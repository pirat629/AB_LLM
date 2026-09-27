import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

EMBED_MODEL, DIM = "openai/text-embedding-3-small", 512

DATA = Path("data")
JOURNAL_PATH = Path("episodes.jsonl")
FACTS_FILE = Path("facts.json")

HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
    "Content-Type": "application/json"
}

MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6"
}
COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}

FACTS_SYSTEM = ("Ты ведёшь память ассистента о пользователе. Верни один JSON-объект вида {\"facts\": [\"...\"]}: полный обновлённый список "
                "коротких фактов о пользователе. Добавь новые устойчивые факты, например: имя, город, работа, интересы, пожелания к ответам, желаемые способы проведения тестов и так далее. "
                "НИКОГДА не добавляй в память пароли, ключи, персональные данные. "
                "Если новый факт отменяет старый, замени старый. Разовые вопросы и мелочи не записывай.")

ASSISTANT_SYSTEM = ("Ты помощник, который отвечает на вопросы по аналитике и проведению АБ тестов. Отвечай на языке пользователя, "
                    "опирайся на источники и на то, что знаешь о пользователе. Источники подобраны автоматически и могут быть не по теме: "
                    "используй только те, что относятся к вопросу. Если в источниках ответа нет, так и скажи.")

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_DB = os.getenv("MILVUS_DB", "")