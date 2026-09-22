from dataclasses import dataclass
from typing import Union
from pydantic import BaseModel, Field

@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list
    cost: float = 0.0
    seconds: float = 0.0

class Answer(BaseModel):
    reasoning: str = Field(description="ход решения в два-три предложения")
    final: Union[str, int, float] = Field(description="только итоговый ответ: число или короткая фраза")