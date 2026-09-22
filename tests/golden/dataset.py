GOLDEN_CASES = [
    {
        "id": "select_criterion",
        "question": "Выбери статистический крпитерий для данных. Группа А: (100, 103, 102), Группа B: (100, 101, 120)",
        "answer_contains": ["ман"],
    },
    {
        "id": "simple_sql_generation",
        "question": "Напиши SQL-запрос для таблицы users (id, age, city). Найди средний возраст юзеров из Москвы.",
        "tools_expected": ["generate_basic_sql"],
        "answer_contains": ["select", "avg", "age", "москва"],
    },
    {
        "id": "ab_test_significance",
        "question": "У нас прошел A/B тест конверсии в покупку. В группе А (0, 1, 0, 0, 1), в группе В (1, 1, 1, 0, 1). Есть ли значимая разница?",
        "answer_contains": ["нет"],
    },
    {
        "id": "no_tools_needed",
        "question": "Сколько будет 2+2? Ответь без использования инструментов.",
        "answer_contains": ["4"],
    },
    {
        "id": "error_recovery",
        "question": "Посчитай A/B тест метрики выручки для групп A (100, 105, 102) и B (110, 115, 112), используя критерий Фишера. Ответь одним словом (Да или нет) стоит ли раскатывать метрики",
        "answer_contains": ["нет"],
    },

]