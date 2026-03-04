from pydantic import BaseModel
from typing import List

import dotenv

import asyncio
from langchain_core.tools import tool

from utils.MyLogs import setup_logger

from Databases.QInteracter import QInteracter
from Databases.NeoInteracter import NeoInteracter
from Handling.Embedder import Embedder


dotenv.load_dotenv()

# Настройка логов
logger = setup_logger(__file__)


class DialogMessage(BaseModel):
    message: str
    role: str

class QueryRequest(BaseModel):
    dialogId: str
    dialogMessages: List[DialogMessage]
    question: str

_embedder = Embedder()
_qdrant = QInteracter()
_neo = NeoInteracter()


@tool
async def rag_tool(question: str, dialog_id: str) -> str:
    """
    Инструмент для получения контекста из RAG системы.
    
    Извлекает ключевые аспекты из вопроса, находит релевантные чанки
    в векторной БД и строит графовый контекст для ответа.
    
    
    Args:
        question: Вопрос пользователя, на который нужно найти контекст
        dialog_id: Идентификатор диалога (указан в системном сообщении как "Current dialog ID")
        
    Returns:
        Контекст в виде текста для генерации ответа
    """
    aspects = []  # список словарей, где каждый словарь - аспект с метаданными

    logger.info("Выделяем аспекты из запроса")
    # выделяем аспекты из запроса
    aspects_text = await _qdrant.extract_aspects_from_question(question)
    for aspect in aspects_text:
        aspects.append({"text": aspect, "dialog_id": dialog_id})

    logger.info("Векторизуем аспекты")
    # векторизуем аспекты
    for aspect in aspects:
        aspect["dense_vector"] = await asyncio.to_thread(_embedder.embed, aspect["text"])

    # [{text: str, dense_vector: list, dialog_id: str}, {}, ...] - aspects сейчас

    logger.info("Поиск по qdrant")
    # dense поиск по qdrant для каждого аспекта
    all_chunks = []  # объединяем все чанки из всех аспектов
    for aspect in aspects:
        chunks = await _qdrant.dense_search(aspect, topk=5)
        all_chunks.extend(chunks)  # добавляем чанки в общий список
    # получаем список всех уникальных чанков из всех аспектов

    logger.info("Поиск по neo4j")
    if all_chunks:
        graph_data = await _neo.graph_context_from_chunks(all_chunks, dialog_id=dialog_id)
        context = graph_data.get("context_text", "")
    else:
        context = ""

    return context

tools = [rag_tool]
tools_by_name = {tool.name: tool for tool in tools}