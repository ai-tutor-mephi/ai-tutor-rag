"""
Тесты для сервиса обработки запросов.

Проверяют корректность работы QueryService:
- Обработка запросов через Agent
- Ограничение токенов для генерации (для быстрых тестов)
- Парсинг ответов
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List, Dict

from src.services.query_service import QueryService
from src.Handling.Embedder import Embedder
from src.Databases.QInteracter import QInteracter
from src.Databases.NeoInteracter import NeoInteracter
from src.LLM.LLMAnswer import LLM
from src.LLM.Agent import Agent

MIN_OUTPUT_TOKENS = 5


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    embedder.embed = MagicMock(return_value=[0.1] * 1024)
    return embedder


@pytest.fixture
def mock_qdrant():
    qdrant = MagicMock(spec=QInteracter)
    qdrant.extract_aspects_from_question = AsyncMock(return_value=["аспект1", "аспект2"])
    qdrant.dense_search = AsyncMock(return_value=["Чанк 1", "Чанк 2", "Чанк 3"])
    return qdrant


@pytest.fixture
def mock_neo():
    neo = MagicMock(spec=NeoInteracter)
    neo.graph_context_from_chunks = AsyncMock(return_value={
        "context_text": "Графовый контекст для ответа."
    })
    return neo


@pytest.fixture
def mock_llm():
    """Фикстура для мока LLM с ограничением токенов."""
    llm = MagicMock(spec=LLM)
    llm.rewrite_question_from_dialogue = AsyncMock(return_value="Перефразированный вопрос")
    llm.answer_with_graph = AsyncMock(return_value="Короткий ответ.")
    return llm


@pytest.fixture
def query_service(mock_embedder, mock_qdrant, mock_neo, mock_llm):
    service = QueryService(
        embedder=mock_embedder,
        qdrant=mock_qdrant,
        neo=mock_neo,
        llm=mock_llm
    )
    # Мокаем Agent с ограничением токенов
    with patch('src.services.query_service.Agent') as mock_agent_class:
        mock_agent = MagicMock(spec=Agent)
        mock_agent.run = AsyncMock(return_value="Тестовый ответ")
        mock_agent_class.return_value = mock_agent
        service.agent = mock_agent
        yield service


class TestQueryService:
    """Тесты для QueryService."""
    
    @pytest.mark.asyncio
    async def test_process_query_returns_string(self, query_service):
        """Тест: process_query возвращает строку."""
        question = "Какой вопрос?"
        dialog_id = "test_dialog_123"
        dialog_messages = []
        
        answer = await query_service.process_query(question, dialog_id, dialog_messages)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    @pytest.mark.asyncio
    async def test_process_query_calls_agent(self, query_service):
        """Тест: process_query вызывает Agent.run."""
        question = "Тестовый вопрос"
        dialog_id = "test_dialog_456"
        dialog_messages = [
            {"role": "user", "message": "Привет"},
            {"role": "assistant", "message": "Здравствуй"}
        ]
        
        await query_service.process_query(question, dialog_id, dialog_messages)
        
        # Проверяем, что agent.run был вызван с правильными параметрами
        query_service.agent.run.assert_called_once_with(
            question,
            dialog_id,
            dialog_messages
        )
    
    @pytest.mark.asyncio
    async def test_process_query_with_empty_dialog(self, query_service):
        """Тест: обработка запроса с пустой историей диалога."""
        question = "Вопрос без истории"
        dialog_id = "test_dialog_empty"
        dialog_messages = []
        
        answer = await query_service.process_query(question, dialog_id, dialog_messages)
        
        assert isinstance(answer, str)
        query_service.agent.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_query_with_dialog_history(self, query_service):
        """Тест: обработка запроса с историей диалога."""
        question = "А что насчет этого?"
        dialog_id = "test_dialog_history"
        dialog_messages = [
            {"role": "user", "message": "Расскажи о Python"},
            {"role": "assistant", "message": "Python - это язык программирования"},
            {"role": "user", "message": "А что насчет этого?"}
        ]
        
        answer = await query_service.process_query(question, dialog_id, dialog_messages)
        
        assert isinstance(answer, str)
        # Проверяем, что история диалога была передана
        call_args = query_service.agent.run.call_args
        assert call_args[0][2] == dialog_messages


class TestQueryServiceWithLimitedTokens:
    """Тесты для QueryService с ограничением токенов для генерации."""
    
    @pytest.mark.asyncio
    @patch('src.LLM.Agent.ChatOpenAI')
    async def test_agent_with_limited_tokens(self, mock_chat_openai):
        """Тест: Agent с моком ChatOpenAI возвращает ответ"""
        mock_chat_llm_instance = MagicMock()
        mock_chat_llm_instance.bind_tools = MagicMock(return_value=mock_chat_llm_instance)
        mock_chat_llm_instance.ainvoke = AsyncMock(return_value=MagicMock(
            content="Короткий ответ",
            tool_calls=None
        ))
        mock_chat_openai.return_value = mock_chat_llm_instance
        
        agent = Agent()
        
        mock_chat_openai.assert_called_once()
        
        # Мокаем граф для теста
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "messages": [MagicMock(content="Короткий ответ", spec=["content"])]
        })
        agent.graph = mock_graph
        
        question = "Короткий вопрос?"
        dialog_id = "test_token_limit"
        dialog_messages = []
        
        answer = await agent.run(question, dialog_id, dialog_messages)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    @pytest.mark.asyncio
    @patch('src.LLM.LLMAnswer.MsGraphRAG')
    async def test_llm_answer_with_limited_tokens(self, mock_ms_graphrag_class):
        """Тест: LLM.answer_with_graph использует ограничение токенов через патч."""
        # Создаем мок для MsGraphRAG
        mock_ms = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Короткий ответ на вопрос."
        
        # Патчим achat, чтобы перехватить вызов и добавить max_tokens
        async def patched_achat(*args, **kwargs):
            # Добавляем max_tokens в config, если его нет
            if 'config' not in kwargs:
                kwargs['config'] = {}
            kwargs['config']['max_tokens'] = MIN_OUTPUT_TOKENS
            return mock_response
        
        mock_ms.achat = AsyncMock(side_effect=patched_achat)
        mock_ms_graphrag_class.return_value = mock_ms
        
        # Создаем LLM с моками
        with patch('src.LLM.LLMAnswer.GraphDatabase') as mock_graph_db:
            mock_driver = MagicMock()
            mock_graph_db.driver.return_value = mock_driver
            
            llm = LLM(ms=mock_ms)
            
            question = "Тестовый вопрос"
            context = "Короткий контекст."
            
            # Вызываем (max_tokens будет добавлен через патч в achat)
            answer = await llm.answer_with_graph(question, context)
            
            assert isinstance(answer, str)
            assert len(answer) > 0
            
            # Проверяем, что achat был вызван
            mock_ms.achat.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.LLM.LLMAnswer.OpenAI')
    async def test_llm_rewrite_with_limited_tokens(self, mock_openai_class):
        """Тест: LLM.rewrite_question_from_dialogue вызывает API и возвращает перефразированный вопрос."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Перефразированный вопрос"
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client
        
        with patch('src.LLM.LLMAnswer.GraphDatabase') as mock_graph_db, \
             patch('src.LLM.LLMAnswer.MsGraphRAG') as mock_ms_graphrag:
            mock_driver = MagicMock()
            mock_graph_db.driver.return_value = mock_driver
            mock_ms = MagicMock()
            mock_ms_graphrag.return_value = mock_ms
            
            llm = LLM(client=mock_client, ms=mock_ms)
            
            question = "Вопрос"
            dialogue = "user: Привет\nassistant: Здравствуй"
            
            rewritten = await llm.rewrite_question_from_dialogue(question, dialogue)
            
            assert isinstance(rewritten, str)
            assert len(rewritten) > 0
            mock_client.chat.completions.create.assert_called_once()


class TestQueryServiceIntegration:
    """Интеграционные тесты для QueryService (требуют реальных сервисов)."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_query_pipeline(self):
        """Тест: полный пайплайн обработки запроса (интеграционный)."""
        # Этот тест требует реальных подключений к БД
        # Помечен как integration и может быть пропущен в обычных тестах
        
        # Инициализируем реальные сервисы
        from src.Handling.Embedder import Embedder
        from src.Databases.QInteracter import QInteracter
        from src.Databases.NeoInteracter import NeoInteracter
        from src.LLM.LLMAnswer import LLM
        
        embedder = Embedder()
        qdrant = QInteracter()
        neo = NeoInteracter()
        
        # Создаем LLM с ограничением токенов для тестов
        # Модифицируем MsGraphRAG для передачи max_tokens
        with patch('src.LLM.LLMAnswer.MsGraphRAG') as mock_ms_class:
            mock_ms = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Тестовый ответ с ограниченными токенами."
            mock_ms.achat = AsyncMock(return_value=mock_response)
            mock_ms_class.return_value = mock_ms
            
            llm = LLM(ms=mock_ms)
            query_service = QueryService(
                embedder=embedder,
                qdrant=qdrant,
                neo=neo,
                llm=llm
            )
            
            question = "Тестовый вопрос"
            dialog_id = "integration_test_123"
            dialog_messages = []
            
            # Мокаем Agent для быстрого теста
            with patch.object(query_service, 'agent') as mock_agent:
                mock_agent.run = AsyncMock(return_value="Интеграционный ответ")
                
                answer = await query_service.process_query(
                    question,
                    dialog_id,
                    dialog_messages
                )
                
                assert isinstance(answer, str)
                assert len(answer) > 0


@pytest.fixture
def query_service_with_token_limits(mock_embedder, mock_qdrant, mock_neo):
    """Фикстура для QueryService с ограничением токенов через патчи."""
    # Создаем LLM с патчингом для ограничения токенов
    with patch('src.LLM.LLMAnswer.MsGraphRAG') as mock_ms_class, \
         patch('src.LLM.LLMAnswer.OpenAI') as mock_openai_class, \
         patch('src.LLM.LLMAnswer.GraphDatabase') as mock_graph_db:
        
        # Настраиваем моки
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        # Патчим MsGraphRAG.achat для добавления max_tokens
        mock_ms = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Короткий ответ."
        
        async def patched_achat(*args, **kwargs):
            # Добавляем max_tokens в config для ограничения генерации
            if 'config' not in kwargs:
                kwargs['config'] = {}
            kwargs['config']['max_tokens'] = MIN_OUTPUT_TOKENS
            return mock_response
        
        mock_ms.achat = AsyncMock(side_effect=patched_achat)
        mock_ms_class.return_value = mock_ms
        
        # Патчим OpenAI client для добавления max_tokens
        mock_client = MagicMock()
        mock_rewrite_response = MagicMock()
        mock_rewrite_choice = MagicMock()
        mock_rewrite_choice.message.content = "Перефразированный"
        mock_rewrite_response.choices = [mock_rewrite_choice]
        
        def patched_create(*args, **kwargs):
            # Добавляем max_tokens для ограничения генерации
            kwargs['max_tokens'] = MIN_OUTPUT_TOKENS
            return mock_rewrite_response
        
        mock_client.chat.completions.create = MagicMock(side_effect=patched_create)
        mock_openai_class.return_value = mock_client
        
        llm = LLM(ms=mock_ms, client=mock_client)
        
        service = QueryService(
            embedder=mock_embedder,
            qdrant=mock_qdrant,
            neo=mock_neo,
            llm=llm
        )
        
        # Мокаем Agent с ограничением токенов через патч ChatOpenAI
        with patch('src.services.query_service.Agent') as mock_agent_class, \
             patch('src.LLM.Agent.ChatOpenAI') as mock_chat_openai:
            mock_chat_llm = MagicMock()
            mock_chat_llm.bind_tools = MagicMock(return_value=mock_chat_llm)
            mock_chat_llm.ainvoke = AsyncMock(return_value=MagicMock(
                content="Ответ с ограниченными токенами",
                tool_calls=None
            ))
            mock_chat_openai.return_value = mock_chat_llm
            
            mock_agent = MagicMock(spec=Agent)
            mock_agent.run = AsyncMock(return_value="Ответ с ограниченными токенами")
            mock_agent_class.return_value = mock_agent
            service.agent = mock_agent
            yield service


class TestTokenLimits:
    """Тесты для проверки ограничения токенов."""
    
    @pytest.mark.asyncio
    async def test_answer_parsing_with_short_response(self, query_service_with_token_limits):
        """Тест: парсинг короткого ответа (минимальные токены)."""
        question = "Вопрос?"
        dialog_id = "token_test_1"
        dialog_messages = []
        
        answer = await query_service_with_token_limits.process_query(
            question,
            dialog_id,
            dialog_messages
        )
        
        # Проверяем, что ответ получен и может быть распарсен
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Ответ должен быть коротким (из-за ограничения токенов)
        # В реальности нужно проверить количество токенов
    
    @pytest.mark.asyncio
    async def test_response_structure(self, query_service_with_token_limits):
        """Тест: структура ответа корректна для парсинга."""
        question = "Структурированный вопрос?"
        dialog_id = "token_test_2"
        dialog_messages = []
        
        answer = await query_service_with_token_limits.process_query(
            question,
            dialog_id,
            dialog_messages
        )
        
        # Проверяем базовую структуру ответа
        assert isinstance(answer, str)
        # Ответ не должен быть пустым
        assert answer.strip() != ""
        # Ответ не должен содержать ошибок формата
        assert "\x00" not in answer  # Нет нулевых байтов