## Запуск тестов

### Из корневой директории проекта (рекомендуется)

```bash
# Все тесты
pytest tests/

# Конкретный файл тестов
pytest tests/test_load.py
pytest tests/test_query.py

# С подробным выводом
pytest tests/ -v

# Только unit-тесты (без интеграционных)
pytest tests/ -m "not integration"
```

### Из директории tests

```bash
cd tests
pytest test_load.py
pytest test_query.py
```

## Запуск с покрытием кода

```bash
# Установите pytest-cov (если еще не установлен)
pip install pytest-cov

# Запустите тесты с покрытием
pytest tests/ --cov=. --cov-report=html

xdg-open htmlcov/index.html  # Linux
```

## Интеграционные тесты

Интеграционные тесты помечены маркером `@pytest.mark.integration` и требуют:
- Запущенные сервисы (Qdrant, Neo4j)
- Настроенные переменные окружения
- Реальные подключения к БД

Для запуска только интеграционных тестов:
```bash
pytest tests/ -m integration
```

Для пропуска интеграционных тестов:
```bash
pytest tests/ -m "not integration"
```
