**Запуск тестов:**
```bash
python -m pytest -m "not golden" -v
```

**Запуск голден запросов**
```bash
python -m pytest -m "golden" -v -s
```

**Скиллы**

Скиллы можно посмотреть по пути **src → tools → agent_skills**

Функция регистрации **src → core → registry**