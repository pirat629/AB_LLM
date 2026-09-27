import pytest
from unittest.mock import patch

from src.tools.agent_skills import (
    generate_basic_sql,
    profile_metric,
    select_stat_criterion,
    calculate_ab_test,
    interpret_results, knowledge_base
)

# Тесты для generate_basic_sql
@patch("src.tools.agent_skills.chat")
def test_generate_basic_sql_normal(mock_chat):
    mock_chat.return_value = {"content": "SELECT SUM(revenue) FROM table_a;"}

    result = generate_basic_sql("Какая выручка?", "table_a (revenue int)")

    assert result == "SELECT SUM(revenue) FROM table_a;"
    mock_chat.assert_called_once()


@patch("src.tools.agent_skills.chat")
def test_generate_basic_sql_empty(mock_chat):
    mock_chat.return_value = {"content": ""}

    result = generate_basic_sql("...", "...")
    assert result == ""


@patch("src.tools.agent_skills.chat")
def test_generate_basic_sql_error(mock_chat):
    mock_chat.side_effect = RuntimeError("HTTP 500: Server Error")

    with pytest.raises(RuntimeError, match="HTTP 500"):
        generate_basic_sql("...", "...")


# Тесты для profile_metric
def test_profile_metric_normal():
    data_a = [0, 1, 0, 1, 1]
    data_b = [1, 1, 1, 0, 1]

    result = profile_metric(data_a, data_b)

    assert result["is_binary"] is True
    assert result["has_outliers"] is False
    assert result["size_a"] == 5


def test_profile_metric_empty():
    with pytest.raises(ValueError, match="Массивы данных не могут быть пустыми"):
        profile_metric([], [])


def test_profile_metric_error():
    with pytest.raises(TypeError):
        profile_metric(["a", "b"], [1, 2])


# Тесты для select_stat_criterion
def test_select_stat_criterion_normal():
    profile = {"is_binary": False, "has_outliers": True}
    assert select_stat_criterion(profile) == "mann_whitney"


def test_select_stat_criterion_edge_case():
    profile = {"is_binary": False, "has_outliers": False}
    assert select_stat_criterion(profile) == "welch_t_test"


def test_select_stat_criterion_error():
    profile = {"is_binary": True}
    profile_fail = {"is_binary": False}
    with pytest.raises(KeyError, match="has_outliers"):
        select_stat_criterion(profile_fail)


# Тесты для calculate_ab_test
def test_calculate_ab_test_normal():
    data_a = [10.5, 12.1, 11.8, 10.9]
    data_b = [14.2, 15.1, 13.8, 14.9]

    result = calculate_ab_test(data_a, data_b, "welch_t_test", alpha=0.05)

    assert result["criterion_used"] == "welch_t_test"
    assert result["p_value"] < 0.05
    assert result["is_significant"] is True
    assert result["relative_diff_pct"] > 0


def test_calculate_ab_test_empty_diff():
    data_a = [10, 10, 10, 10]
    data_b = [10, 10, 10, 10]

    result = calculate_ab_test(data_a, data_b, "welch_t_test")

    assert result["is_significant"] is False
    assert result["relative_diff_pct"] == 0.0


def test_calculate_ab_test_error():
    with pytest.raises(ValueError, match="Неизвестный критерий"):
        calculate_ab_test([1, 2], [1, 2], "unknown_magic_test")


# Тесты для interpret_results
@patch("src.tools.agent_skills.chat")
def test_interpret_results_normal(mock_chat):
    mock_chat.return_value = {"content": "Раскатываем фичу, эффект выше MDE."}

    test_res = {
        "criterion_used": "welch_t_test",
        "p_value": 0.01,
        "alpha": 0.05,
        "is_significant": True,
        "relative_diff_pct": 5.5
    }

    result = interpret_results(test_res, mde_pct=2.0)
    assert result == "Раскатываем фичу, эффект выше MDE."
    assert "2.0%" in mock_chat.call_args[0][0][0]["content"]


@patch("src.tools.agent_skills.chat")
def test_interpret_results_empty_mde(mock_chat):
    mock_chat.return_value = {"content": "Вывод без учета MDE"}

    test_res = {
        "criterion_used": "z_test_proportions",
        "p_value": 0.8,
        "alpha": 0.05,
        "is_significant": False,
        "relative_diff_pct": 0.1
    }

    result = interpret_results(test_res, mde_pct=None)
    assert result == "Вывод без учета MDE"
    assert "Минимальный ожидаемый эффект" not in mock_chat.call_args[0][0][0]["content"]


@patch("src.tools.agent_skills.chat")
def test_interpret_results_error(mock_chat):
    bad_test_res = {"p_value": 0.05}

    with pytest.raises(KeyError):
        interpret_results(bad_test_res)


#knowledge_base
@patch("src.tools.agent_skills.search_milvus")
def test_knowledge_base_normal(mock_search_milvus):
    mock_search_milvus.return_value = [
        {"H1": "Аналитика", "H2": "Линеаризация", "text": "Дельта-метод для ratio метрик."},
        {"H1": "Статистика", "text": "Описание дисперсии."}
    ]

    result = knowledge_base(query="delta method", page="Линеаризация")

    assert "[Аналитика > Линеаризация] Дельта-метод для ratio метрик." in result
    assert "[Статистика] Описание дисперсии." in result

    mock_search_milvus.assert_called_once()
    call_args = mock_search_milvus.call_args[0]
    assert call_args[0] == "lines"
    assert call_args[1] == "delta method"
    assert call_args[2] == 5
    assert 'metadata["H1"] like "%Линеаризация%"' in call_args[3]


@patch("src.tools.agent_skills.search_milvus")
def test_knowledge_base_empty(mock_search_milvus):
    mock_search_milvus.return_value = []

    result = knowledge_base(query="unknown abstract query")

    assert result == "nothing found"

    call_args = mock_search_milvus.call_args[0]
    assert call_args[3] == ""


@patch("src.tools.agent_skills.search_milvus")
def test_knowledge_base_error(mock_search_milvus):
    bad_hits = [
        {"H1": "Аналитика", "H2": "CUPED"}
    ]
    mock_search_milvus.return_value = bad_hits

    with pytest.raises(KeyError):
        knowledge_base(query="cuped variance reduction")