from src.config import MODELS
from src.core.agent import agent
from src.core.ledger import ledger
from src.utils.trace import show_trace
import src.tools.agent_skills
from src.core.registry import TOOLS

if __name__ == "__main__":
    question = "Посчитай A/B тест метрики выручки для групп A (100, 105, 102) и B (110, 115, 112), используя критерий Фишера. Ответь одним словом (Да или нет) стоит ли раскатывать метрики"

    run = agent(
        question=question,
        model=MODELS["cheap"],
        tool_names=list(TOOLS.keys()),
    )

    show_trace(run)
    print("\nФинальный ответ:", run.answer)
    print("\n--- Затраты ---")
    print(ledger.table())