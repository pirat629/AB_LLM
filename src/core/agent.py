import json
import re
import time
from typing import Type
from pydantic import BaseModel

from src.core.schemas import Run, Answer
from src.core.registry import TOOLS
from src.core.ledger import ledger
from src.core.llm import chat

SYSTEM = ("Ты решаешь задачи. Если нужно посчитать или найти факт, вызывай инструменты, а не угадывай. "
          "Когда ответ готов, напиши его строго одним JSON-объектом по этой схеме, без текста вокруг:\n{template}\n"
          "числа и фразы пиши в строковом формате")


def json_from(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("В ответе нет JSON")
    return json.loads(m.group(0), strict=False)


def get_schema_template(schema):
    template = {k: v.description for k, v in schema.model_fields.items()}
    return json.dumps(template, ensure_ascii=False, indent=2)


def run_tool(call):
    name = call["function"]["name"]
    try:
        spec = TOOLS[name]
        args = spec["args"].model_validate_json(call["function"]["arguments"])
        result = spec["fn"](**args.model_dump())
    except Exception as e:
        result = f"Не удалось вызвать инструмент {name}: {e}"
    return {"role": "tool", "tool_call_id": call["id"], "content": str(result)[:2000]}


def looped(seen, calls):
    keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    repeated = any(k in seen for k in keys)
    seen.update(keys)
    return repeated


def finish(model, messages, step):
    del messages[-1]
    messages.append({"role": "user", "content": "Инструменты больше недоступны. Ответь по тому, что известно."})
    msg = chat(messages, model, tag="agent")
    messages.append(msg)
    return json_from(msg.get("content", "")), step + 1


def agent_loop(messages, model, tool_names, max_steps):
    seen = set()
    for step in range(1, max_steps + 1):
        active_tools = [TOOLS[n]["schema"] for n in tool_names if n in TOOLS]
        msg = chat(messages, model, tools=active_tools or None, tag="agent")
        messages.append(msg)
        calls = msg.get("tool_calls", [])

        if not calls:
            return json_from(msg.get("content", "")), step
        if looped(seen, calls) or step == max_steps:
            return finish(model, messages, step)

        messages += [run_tool(c) for c in calls]
    return None


def agent(question, model, tool_names, max_steps=8, schema: Type[BaseModel] = Answer):
    json_template = get_schema_template(schema)
    messages = [
        {"role": "system", "content": SYSTEM.format(template=json_template)},
        {"role": "user", "content": question}
    ]

    before, started = ledger.total, time.perf_counter()
    answer, steps = agent_loop(messages, model, tool_names, max_steps)

    return Run(question, answer, steps, messages, ledger.total - before, time.perf_counter() - started)