"""
FastAPI приложение для RAG (Retrieval-Augmented Generation) системы.

Этот модуль содержит только HTTP эндпоинты. Вся бизнес-логика вынесена
в отдельные сервисы (services/load_service.py и services/query_service.py).

Архитектура:
- Эндпоинты (этот файл) - только валидация запросов и вызов сервисов
- Сервисы (services/) - бизнес-логика и оркестрация пайплайна
- Классы (Handling/, Databases/, LLM/) - детальная реализация операций
"""

import logging

import dotenv

dotenv.load_dotenv()

from logging_setup import configure_logging

configure_logging()

import fastapi
from fastapi.responses import JSONResponse

from src.utils.ragPydantic import LoadRequest, QueryRequest, TestsRequest

# Импорт сервисов для бизнес-логики
from src.services import LoadService, QueryService

# Импорт классов для инициализации сервисов
from src.Handling.Embedder import Embedder
from src.Handling.Chunker import Chunker
from src.Databases.QInteracter import QInteracter
from src.Databases.NeoInteracter import NeoInteracter
from src.LLM.LLMAnswer import LLM
from src.services.test_generation_service import generate_tests

logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
rag = fastapi.FastAPI(
    title="RAG Service",
    description="Сервис для загрузки документов и генерации ответов на основе RAG",
    version="1.0.0"
)

# Инициализация компонентов системы
# Эти экземпляры используются всеми сервисами
qdrant = QInteracter()
neo = NeoInteracter()
llm = LLM()
embedder = Embedder()
chunker = Chunker()

# Инициализация сервисов бизнес-логики
load_service = LoadService(
    chunker=chunker,
    embedder=embedder,
    qdrant=qdrant,
    neo=neo
)

query_service = QueryService(
    embedder=embedder,
    qdrant=qdrant,
    neo=neo,
    llm=llm
)


@rag.post("/tests")
async def create_tests(data: TestsRequest) -> JSONResponse:
    """
    Эндпоинт для создания тестов для диалога.
    
    Принимает идентификатор диалога и историю диалога и создает тесты для диалога.
    """
    try:
        logger.info("Получен запрос на создание тестов для диалога")
        data_dict = data.model_dump()
        dialog_id = data_dict['dialogId']
        dialog_messages = data_dict['dialogMessages']
    except Exception as e:
        logger.error("Не корректный запрос: %s", e)
        return JSONResponse(
            content={'message': f'Не корректный запрос:\n{e}'},
            status_code=400
        )
    
    try:
        tests = await generate_tests(dialog_messages, neo, dialog_id)
    except Exception as e:
        logger.error("Ошибка при создании тестов для диалога: %s", e)
        return JSONResponse(
            content={'message': f'Не удалось создать тесты для диалога:\n{e}'},
            status_code=500
        )
    
    return JSONResponse(content=tests, status_code=200)


@rag.post("/load")
async def load_endpoint(data: LoadRequest) -> JSONResponse:
    """
    Эндпоинт для загрузки документов в систему.
    
    Принимает список файлов и загружает их в базы данных:
    - Qdrant (векторная БД для семантического поиска)
    - Neo4j (графовая БД для знаний)
    
    Пайплайн обработки (скрыт в LoadService):
    1. Разбиение текста на чанки
    2. Векторизация чанков
    3. Сохранение в Qdrant
    4. Создание графа в Neo4j
    
    Request Body:
        {
            "content": [
                {
                    "fileId": "идентификатор_документа",
                    "fileName": "название_файла",
                    "text": "текст_документа"
                }
            ],
            "dialogId": "идентификатор_диалога"
        }
    
    Returns:
        JSONResponse с результатом загрузки
    """
    try:
        # Валидация и преобразование данных
        logger.info("Получен запрос на загрузку документов")
        data_dict = data.model_dump()
        content = data_dict['content']
        dialog_id = data_dict['dialogId']
        
        logger.info("Обработка %s файлов для диалога %s", len(content), dialog_id)
        
        # Вызов сервиса для обработки файлов
        await load_service.process_files(
            content=content,
            dialog_id=dialog_id
        )
        
        logger.info("Документы успешно загружены для диалога %s", dialog_id)
        return JSONResponse(
            content={'message': 'OK'},
            status_code=200
        )
        
    except Exception as e:
        logger.error("Ошибка при загрузке документов: %s", e)
        return JSONResponse(
            content={'message': f'Не удалось загрузить документы:\n{e}'},
            status_code=500
        )


@rag.post("/query")
async def query_endpoint(data: QueryRequest) -> JSONResponse:
    """
    Эндпоинт для получения ответа на вопрос пользователя.
    
    Обрабатывает вопрос пользователя и возвращает ответ на основе:
    - Загруженных документов (через векторный поиск в Qdrant)
    - Графа знаний (через поиск в Neo4j)
    - Контекста диалога (для перефразирования вопроса)
    
    Пайплайн обработки (скрыт в QueryService):
    1. Перефразирование вопроса на основе истории диалога
    2. Извлечение ключевых аспектов из вопроса
    3. Векторный поиск релевантных чанков в Qdrant
    4. Построение графового контекста в Neo4j
    5. Генерация ответа через LLM
    
    Request Body:
        {
            "dialogId": "идентификатор_диалога",
            "dialogMessages": [
                {
                    "message": "текст_сообщения",
                    "role": "user|assistant"
                }
            ],
            "question": "вопрос_пользователя"
        }
    
    Returns:
        JSONResponse с ответом и идентификатором диалога
    """
    try:
        # Валидация и преобразование данных
        logger.info("Получен запрос на обработку вопроса")
        data_dict = data.model_dump()
        dialog_id = data_dict["dialogId"]
        dialog_messages = data_dict["dialogMessages"]
        question = data_dict["question"]
        
        logger.info("Обработка вопроса для диалога %s: %s...", dialog_id, question[:100])
        
        # Вызов сервиса для обработки запроса
        # Вся бизнес-логика скрыта внутри сервиса
        answer = await query_service.process_query(
            question=question,
            dialog_id=dialog_id,
            dialog_messages=dialog_messages
        )
        
        logger.info("Ответ сгенерирован для диалога %s", dialog_id)
        return JSONResponse(
            content={
                'answer': answer,
                'dialogId': dialog_id
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error("Ошибка при обработке запроса: %s", e)
        return JSONResponse(
            content={'message': f'Ошибка при обработке запроса:\n{e}'},
            status_code=500
        )
