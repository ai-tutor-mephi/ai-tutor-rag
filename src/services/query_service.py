"""
Сервис для обработки запросов и генерации ответов.

Этот модуль содержит всю бизнес-логику обработки запросов пользователя:
1. Перефразирование вопроса на основе истории диалога
2. Извлечение аспектов из вопроса
3. Векторный поиск в Qdrant
4. Поиск контекста в графе Neo4j
5. Генерация ответа через LLM
"""

import asyncio
import logging
from typing import List, Dict

from ..Handling.Embedder import Embedder
from ..Databases.QInteracter import QInteracter
from ..Databases.NeoInteracter import NeoInteracter
from ..LLM.LLMAnswer import LLM
from ..LLM.Agent import Agent

logger = logging.getLogger(__name__)


class QueryService:
    """
    Сервис для обработки запросов пользователя и генерации ответов.
    
    Реализует полный пайплайн RAG:
    - Перефразирование вопроса с учетом контекста диалога
    - Извлечение ключевых аспектов
    - Поиск релевантных чанков в векторной БД
    - Построение графового контекста
    - Генерация финального ответа
    """
    
    def __init__(
        self,
        embedder: Embedder,
        qdrant: QInteracter,
        neo: NeoInteracter,
        llm: LLM
    ):
        """
        Инициализация сервиса обработки запросов.
        
        Args:
            embedder: Сервис для векторизации текста
            qdrant: Клиент для работы с векторной БД Qdrant
            neo: Клиент для работы с графовой БД Neo4j
            llm: Сервис для генерации ответов через LLM
        """
        self.embedder = embedder
        self.qdrant = qdrant
        self.neo = neo
        self.llm = llm
        self.agent = Agent()
    
    async def process_query(
        self,
        question: str,
        dialog_id: str,
        dialog_messages: List[Dict]
    ) -> str:
        """
        Обрабатывает запрос пользователя и возвращает ответ.
        
        Пайплайн обработки:
        1. Перефразирование вопроса на основе истории диалога
        2. Извлечение ключевых аспектов из вопроса
        3. Векторизация аспектов
        4. Поиск релевантных чанков в Qdrant для каждого аспекта
        5. Построение графового контекста в Neo4j
        6. Генерация финального ответа через LLM
        
        Args:
            question: Вопрос пользователя
            dialog_id: Идентификатор диалога
            dialog_messages: История диалога (список сообщений с role и message)
            
        Returns:
            Ответ на вопрос пользователя
        """
        logger.info(f"Обработка запроса для диалога {dialog_id}")
        
        return await self.agent.run(question, dialog_id, dialog_messages)
    
    async def _rewrite_question(
        self,
        question: str,
        dialog_messages: List[Dict]
    ) -> str:
        """
        Перефразирует вопрос с учетом истории диалога.
        
        Это позволяет улучшить понимание вопроса, учитывая контекст
        предыдущих сообщений в диалоге.
        
        Args:
            question: Исходный вопрос пользователя
            dialog_messages: История диалога
            
        Returns:
            Перефразированный вопрос
        """
        logger.info("Перефразирование вопроса на основе истории диалога...")
        
        # Формируем строку диалога из истории
        dialogue_messages = [
            f"{msg['role']}: {msg['message']}"
            for msg in dialog_messages
        ]
        dialogue = ''.join(dialogue_messages)
        
        # Перефразируем вопрос через LLM
        rewritten_question = await self.llm.rewrite_question_from_dialogue(
            question=question,
            dialogue=dialogue
        )
        
        return rewritten_question
    
    async def _extract_aspects(
        self,
        question: str,
        dialog_id: str
    ) -> List[Dict]:
        """
        Извлекает ключевые аспекты из вопроса.
        
        Аспекты - это ключевые темы или концепции, которые нужно
        найти в базе знаний для ответа на вопрос.
        
        Args:
            question: Вопрос (желательно перефразированный)
            dialog_id: Идентификатор диалога
            
        Returns:
            Список словарей с аспектами, каждый содержит:
            - text: текст аспекта
            - dialog_id: идентификатор диалога
            - dense_vector: векторное представление (добавляется позже)
        """
        logger.info("Извлечение аспектов из вопроса...")
        
        # Извлекаем аспекты через LLM
        aspects_text = await self.qdrant.extract_aspects_from_question(question)
        
        # Формируем структуру аспектов
        aspects = []
        for aspect_text in aspects_text:
            aspects.append({
                "text": aspect_text,
                "dialog_id": dialog_id
            })
        
        return aspects
    
    async def _vectorize_aspects(self, aspects: List[Dict]) -> None:
        """
        Векторизует все аспекты, добавляя dense_vector к каждому.
        
        Args:
            aspects: Список аспектов для векторизации (изменяется in-place)
        """
        logger.info("Векторизация аспектов...")
        
        for aspect in aspects:
            aspect["dense_vector"] = await asyncio.to_thread(
                self.embedder.embed,
                aspect["text"]
            )
    
    async def _search_relevant_chunks(
        self,
        aspects: List[Dict]
    ) -> List[List[str]]:
        """
        Ищет релевантные чанки в Qdrant для каждого аспекта.
        
        Для каждого аспекта выполняется векторный поиск в Qdrant,
        который возвращает top-k наиболее похожих чанков.
        
        Args:
            aspects: Список векторизованных аспектов
            
        Returns:
            Список списков, где каждый внутренний список содержит
            тексты релевантных чанков для соответствующего аспекта
        """
        logger.info("Поиск релевантных чанков в Qdrant...")
        
        # Векторизуем аспекты перед поиском
        await self._vectorize_aspects(aspects)
        
        # Ищем релевантные чанки для каждого аспекта
        closest_chunks = []
        for aspect in aspects:
            chunks = await self.qdrant.dense_search(aspect, topk=5)
            closest_chunks.append(chunks)
        
        return closest_chunks
    
    async def _build_graph_context(
        self,
        relevant_chunks: List[List[str]],
        dialog_id: str
    ) -> str:
        """
        Строит графовый контекст на основе найденных чанков.
        
        Для каждой группы релевантных чанков:
        1. Извлекаются сущности из текстов
        2. Находится соответствующий подграф в Neo4j
        3. Формируется текстовое представление контекста
        
        Args:
            relevant_chunks: Список списков релевантных чанков
            dialog_id: Идентификатор диалога
            
        Returns:
            Объединенный графовый контекст в виде текста
        """
        logger.info("Построение графового контекста в Neo4j...")
        
        context = ""
        for chunks in relevant_chunks:
            # Получаем графовый контекст для группы чанков
            graph_data = await self.neo.graph_context_from_chunks(
                chunks,
                dialog_id=dialog_id
            )
            
            # Извлекаем текстовое представление контекста
            context_text = graph_data.get("context_text", "")
            context += context_text
        
        return context
    
    async def _generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Генерирует финальный ответ на основе вопроса и контекста.
        
        Использует LLM для генерации ответа, который учитывает:
        - Исходный вопрос (или перефразированный)
        - Графовый контекст из Neo4j
        
        Args:
            question: Вопрос пользователя
            context: Графовый контекст из Neo4j
            
        Returns:
            Сгенерированный ответ
        """
        logger.info("Генерация ответа через LLM...")
        
        answer = await self.llm.answer_with_graph(question, context)
        
        return answer
