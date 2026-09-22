import pytest
from src.core.agent import agent
from src.config import MODELS
from tests.golden.dataset import GOLDEN_CASES

from src.core.registry import TOOLS
import src.tools

@pytest.mark.golden
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c["id"])
def test_agent_golden_behavior(case):
    run = agent(
        question=case["question"],
        model=MODELS["cheap"],
        tool_names=list(TOOLS.keys()),
        max_steps=case.get("max_steps", 8)
    )

    assert run.answer is not None, "Агент вернул пустой ответ (ошибка парсинга JSON)"

    final_text = str(run.answer.get("final", "")).lower()
    reasoning_text = str(run.answer.get("reasoning", "")).lower()
    combined_text = final_text + " " + reasoning_text

    for substring in case.get("answer_contains", []):
        assert substring.lower() in combined_text, f"Ожидалась подстрока '{substring}' в ответе агента. Ответ: {combined_text}"