import time
import requests
from src.config import CHAT_URL, HEADERS
from src.core.ledger import ledger

def post_with_retry(body, attempts=3, delay=5):
    for attempt in range(attempts):
        response = requests.post(CHAT_URL, json=body, headers=HEADERS, timeout=120)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(delay)
            continue
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:5000]}")
    return None


def chat(messages, model, tools=None, tag="chat", temperature=None):
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature

    started = time.perf_counter()
    data = post_with_retry(body)

    ledger.add(tag, model, data.get("usage", {}), time.perf_counter() - started)
    return data["choices"][0]["message"]