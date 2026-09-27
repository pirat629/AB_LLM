import uuid
from src.core.vectorstore import load_cache, save_cache
from src.core.agent import talk, end_session
from src.core.ledger import ledger

def main():
    print("Инициализация ИИ-агента...")

    cached_vectors = load_cache()
    print(f"Кэш загружен (векторов: {cached_vectors})")

    session = str(uuid.uuid4())
    history = []

    print("Агент готов к работе (введите 'exit' или 'выход' для завершения)\n")
    print("-" * 50)

    try:
        while True:
            user_text = input("Вы: ").strip()

            if not user_text:
                continue

            if user_text.lower() in ["exit", "выход", "quit"]:
                break

            answer = talk(session, history, user_text, mode="память")
            print(f"\nАгент: {answer}\n")
            print("-" * 50)

    finally:
        print("\nЗавершение сессии...")

        if history:
            print("Анализ диалога и извлечение новых фактов...")
            end_session(history)

        print("Сохранение кэша векторов на диск...")
        save_cache()

        print("\nСтатистика затрат на LLM:")
        try:
            print(ledger.table())
            print(f"\nИтого потрачено за сессию: ${ledger.mine:.4f}")
        except Exception as e:
            print(f"Нет данных по затратам или ошибка группировки: {e}")


if __name__ == "__main__":
    main()