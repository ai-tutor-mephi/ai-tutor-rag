"""
Конфигурация для тестов pytest.

Содержит общие фикстуры и настройки для всех тестов.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Добавляем корневую директорию проекта в sys.path для импортов
# Это позволяет запускать тесты из любой директории
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Устанавливаем переменные окружения для тестов, чтобы избежать подключений к БД
# Neo4j и Qdrant будут мокироваться в тестах
if not os.getenv("NEO4J_URI"):
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
if not os.getenv("NEO4J_USERNAME"):
    os.environ["NEO4J_USERNAME"] = "neo4j"
if not os.getenv("NEO4J_PASSWORD"):
    os.environ["NEO4J_PASSWORD"] = "test_password"
if not os.getenv("QDRANT_URL"):
    os.environ["QDRANT_URL"] = "http://localhost:6333"
if not os.getenv("QDRANT_KEY"):
    os.environ["QDRANT_KEY"] = "test_key"

# Минимальные значения токенов для тестов генерации
MIN_INPUT_TOKENS = 10
MIN_OUTPUT_TOKENS = 5


# Патчим подключения к БД ДО импорта любых модулей
# Это критически важно, так как NeoInteracter создает подключения при импорте
import unittest.mock

# Создаем моки
_mock_driver = MagicMock()
_mock_ms_graph = MagicMock()
_mock_ms_graph.query = MagicMock(return_value=[])
_mock_ms_graph.close = MagicMock()
_mock_qdrant_client = MagicMock()

# Применяем патчи на уровне модулей до их импорта
# Патчим neo4j.GraphDatabase
_neo4j_patcher = unittest.mock.patch('neo4j.GraphDatabase')
_neo4j_mock_class = _neo4j_patcher.start()
_neo4j_mock_class.driver = MagicMock(return_value=_mock_driver)

# Патчим ms_graphrag_neo4j.MsGraphRAG  
_ms_graphrag_patcher = unittest.mock.patch('ms_graphrag_neo4j.MsGraphRAG')
_ms_graphrag_mock_class = _ms_graphrag_patcher.start()
_ms_graphrag_mock_class.return_value = _mock_ms_graph

# Патчим qdrant_client.QdrantClient
_qdrant_patcher = unittest.mock.patch('qdrant_client.QdrantClient')
_qdrant_mock_class = _qdrant_patcher.start()
_qdrant_mock_class.return_value = _mock_qdrant_client


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Настройка тестового окружения.
    
    Выполняется один раз для всех тестов.
    """
    # Устанавливаем тестовые переменные окружения
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test_key"
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    if not os.getenv("MS_GRAPHRAG_MODEL"):
        os.environ["MS_GRAPHRAG_MODEL"] = "gpt-4o-mini"
    if not os.getenv("MS_LIGHT_MODEL"):
        os.environ["MS_LIGHT_MODEL"] = "gpt-4o-mini"
    
    yield
    
    pass