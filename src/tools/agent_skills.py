from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.core.registry import register

from src.core.llm import chat
from src.config import MODELS
from src.core.vectorstore import search_milvus


class GenerateBasicSQLArgs(BaseModel):
    question: str = Field(
        description="Бизнес-вопрос аналитику или менеджеру, который нужно перевести в SQL (например: 'Какая выручка была в группе А?')"
    )
    schema_ddl: str = Field(
        description="Строка с DDL (описанием схемы таблиц) базы данных."
    )

def generate_basic_sql(question: str, schema_ddl: str) -> str:
    prompt = (
        f"Ты Data Engineer. Напиши SQL-запрос для PostgreSQL.\n"
        f"Схема БД: {schema_ddl}\n"
        f"Бизнес-вопрос: {question}\n"
        f"Верни только SQL-код без маркдауна."
    )
    response = chat([{"role": "user", "content": prompt}], MODELS['cheap'])
    return response['content']

register(generate_basic_sql,GenerateBasicSQLArgs, "Формирует промпт для перевода бизнес-вопроса в SQL.")

class ProfileMetricArgs(BaseModel):
    data_a: List[float] = Field(
        description="Массив числовых значений метрики для контрольной группы (A). Если значения бинарные, передавать 0 и 1."
    )
    data_b: List[float] = Field(
        description="Массив числовых значений метрики для тестовой группы (B). Если значения бинарные, передавать 0 и 1."
    )

def profile_metric(data_a: list, data_b: list) -> dict:

    if not len(data_a) or not len(data_b):
        raise ValueError("Массивы данных не могут быть пустыми.")

    a, b = np.array(data_a), np.array(data_b)

    is_binary = set(np.unique(a)).issubset({0, 1}) and set(np.unique(b)).issubset({0, 1})

    has_outliers = False
    if not is_binary:
        threshold = np.percentile(a, 99)
        has_outliers = bool(np.sum(a > threshold) > 0)

    return {
        "is_binary": is_binary,
        "has_outliers": has_outliers,
        "size_a": len(a),
        "size_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "var_a": float(np.var(a)),
        "var_b": float(np.var(b)),
    }

register(profile_metric, ProfileMetricArgs, "Считает метаинформацию о распределениях для выбора критерия.")

class SelectStatCriterionArgs(BaseModel):
    profile: Dict[str, Any] = Field(
        description="Словарь с метаинформацией о распределениях данных (результат работы инструмента profile_metric)."
    )

def select_stat_criterion(profile: dict) -> str:
    if profile["is_binary"]:
        return "z_test_proportions"

    if profile["has_outliers"]:
        return "mann_whitney"

    return "welch_t_test"

register(select_stat_criterion, SelectStatCriterionArgs, "Выбор теста в зависимости от метаинформации о распределениях")

class CalculateABTestArgs(BaseModel):
    data_a: List[float] = Field(
        description="Массив числовых значений метрики для контрольной группы (A)."
    )
    data_b: List[float] = Field(
        description="Массив числовых значений метрики для тестовой группы (B)."
    )
    criterion: str = Field(
        description="Название выбранного статистического критерия (результат работы инструмента select_stat_criterion)."
    )
    alpha: float = Field(
        default=0.05,
        description="Уровень статистической значимости (alpha). По умолчанию 0.05."
    )

def calculate_ab_test(data_a: list, data_b: list, criterion: str, alpha: float = 0.05) -> dict:
    a, b = np.array(data_a), np.array(data_b)

    if criterion == "z_test_proportions":
        successes = np.array([np.sum(a), np.sum(b)])
        nobs = np.array([len(a), len(b)])
        stat, p_value = proportions_ztest(successes, nobs)

    elif criterion == "welch_t_test":
        stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    elif criterion == "mann_whitney":
        stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
    else:
        raise ValueError(f"Неизвестный критерий: {criterion}")

    return {
        "criterion_used": criterion,
        "statistic": float(stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_significant": bool(p_value < alpha),
        "relative_diff_pct": float((np.mean(b) - np.mean(a)) / np.mean(a)) * 100
    }

register(calculate_ab_test, CalculateABTestArgs, "Выполнение расчетов A/B теста")

class InterpretResultsArgs(BaseModel):
    test_results: Dict[str, Any] = Field(
        description="Словарь с сухими результатами вычисления A/B теста (результат работы calculate_ab_test)."
    )
    mde_pct: Optional[float] = Field(
        default=None,
        description="Минимально ожидаемый эффект (MDE) в процентах, который хотел увидеть бизнес. Если неизвестен, передать null."
    )

def interpret_results(test_results: dict, mde_pct: float = None) -> str:
    prompt = (
        f"Ты аналитик. Сделай вывод по результатам A/B теста.\n"
        f"Метод: {test_results['criterion_used']}\n"
        f"p-value: {test_results['p_value']:.4f} (при alpha={test_results['alpha']})\n"
        f"Относительное изменение: {test_results['relative_diff_pct']:.2f}%\n"
    )

    if test_results['is_significant']:
        prompt += "Результат: Статистически значимая разница ЕСТЬ.\n"
    else:
        prompt += "Результат: Статистически значимой разницы НЕТ.\n"

    if mde_pct:
        prompt += f"Минимальный ожидаемый эффект (MDE) был: {mde_pct}%\n"

    prompt += "\nНапиши бизнес-вывод: стоит ли раскатывать новую фичу и какие есть риски?"

    response = chat([{"role": "user", "content": prompt}], MODELS['cheap'])
    return response['content']

register(interpret_results, InterpretResultsArgs, "Формулировка итоговых рекомендаций")

class KBArgs(BaseModel):
    query: str = Field(description="Подробный, семантически полный поисковой запрос. Лучше всего передавать переформулированный вопрос пользователя целиком, не сокращая его до ключевых слов.")
    page: str = Field(default="", description="ОПАСНО: Оставляй пустым в 99% случаев! Заполняй только если пользователь ЯВНО просит искать в конкретной статье (например, «Посмотри в статье CUPED»). Регистр имеет значение.")

def knowledge_base(query: str, page: str = "") -> str:
    hits = search_milvus("lines", query, 5, f'metadata["H1"] like "%{page}%" or metadata["source_file"] like "%{page}%"' if page else "")
    if not hits:
        return "Ничего не найдено"
    return "\n".join(f"[{' > '.join(h[f'H{i}'] for i in range(1, 10) if f'H{i}' in h)}] {h['text']}" for h in hits)

register(knowledge_base, KBArgs, "КРИТИЧЕСКИ ВАЖНО: Всегда используй этот инструмент для поиска фактической информации, формул и теории по аналитике. Возвращает 5 релевантных фактов. Не полагайся на свою память.")