from pydantic import BaseModel
from typing import List

import dotenv

import asyncio
from langchain_core.tools import tool

from utils.MyLogs import setup_logger


dotenv.load_dotenv()

# Настройка логов
logger = setup_logger(__name__, __name__)


class DialogMessage(BaseModel):
    message: str
    role: str

class QueryRequest(BaseModel):
    dialogId: str
    dialogMessages: List[DialogMessage]
    question: str


@tool
async def retrieve_context (question:str, dialog_id:str, embedder, qdrant, neo) -> str:
        """
        Получение контекста для вопроса из векторной базы и графовой базы данных
        Если нужно вычленение аспектов из вопроса, раскомментировать соответствующий код
        """

        # aspects = []  # список словарей, где каждый словарь - аспект с метаданными

        # logger.info("Выделяем аспекты из запроса")
        # # выделяем аспекты из запроса
        # aspects_text = await qdrant.extract_aspects_from_question(question)
        # for aspect in aspects_text:
        #     aspects.append({"text": aspect, "dialog_id": dialog_id})

        # logger.info("Векторизуем аспекты")
        # # векторизуем аспекты
        # for aspect in aspects:
        #     aspect["dense_vector"] = await asyncio.to_thread(embedder.embed, aspect["text"])

        # # [{text: str, dense_vector: list, dialog_id: str}, {}, ...] - aspects сейчас

        # logger.info("Поиск по qdrant")
        # # dense поиск по qdrant
        # closest_chunks = []
        # for aspect in aspects:
        #     chunk = await qdrant.dense_search(aspect, topk=5)
        #     closest_chunks.append(chunk)
        # # получаем список списков, где каждый список - тексты ближайших чанков для каждого аспекта


        logger.info("Векторизация вопроса")
        question_vector = await asyncio.to_thread(embedder.embed, question)

        logger.info("Поиск по qdrant")
        closest_chunks = await qdrant.dense_search({"dense_vector": question_vector, "dialog_id": dialog_id}, topk=10)


        logger.info("Поиск по neo4j")
        # поиск по neo4j. Получаем контекст
        context = ""
        for chunks in closest_chunks:
            ctx = await neo.graph_context_from_chunks(chunks, dialog_id=dialog_id)
            context += ctx.get("context_text", "")

        return context

tools = [retrieve_context]
tools_by_name = {tool.name: tool for tool in tools}
