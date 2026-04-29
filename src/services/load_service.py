"""
Сервис для загрузки документов в систему RAG.

Этот модуль содержит всю бизнес-логику обработки и загрузки документов:
1. Разбиение текста на чанки
2. Векторизация чанков
3. Сохранение в векторную БД (Qdrant)
4. Создание графа знаний в Neo4j
"""

import asyncio
import uuid
import logging
from typing import List, Dict

from ..Handling.Chunker import Chunker
from ..Handling.Embedder import Embedder
from ..Databases.QInteracter import QInteracter
from ..Databases.NeoInteracter import NeoInteracter

logger = logging.getLogger(__name__)


class LoadService:
    """
    Сервис для загрузки документов в систему RAG.
    
    Обрабатывает документы, разбивает их на чанки, векторизует
    и сохраняет в базы данных (Qdrant и Neo4j).
    """
    
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        qdrant: QInteracter,
        neo: NeoInteracter
    ):
        """
        Инициализация сервиса загрузки.
        
        Args:
            chunker: Сервис для разбиения текста на чанки
            embedder: Сервис для векторизации текста
            qdrant: Клиент для работы с векторной БД Qdrant
            neo: Клиент для работы с графовой БД Neo4j
        """
        self.chunker = chunker
        self.embedder = embedder
        self.qdrant = qdrant
        self.neo = neo
    
    async def process_file(
        self,
        file_id: str,
        file_name: str,
        text: str,
        dialog_id: str
    ) -> None:
        """
        Обрабатывает один файл: разбивает на чанки, векторизует и загружает в БД.
        
        Пайплайн обработки:
        1. Разбиение текста на чанки
        2. Векторизация каждого чанка
        3. Загрузка в Qdrant (векторная БД)
        4. Загрузка в Neo4j (граф знаний)
        
        Args:
            file_id: Уникальный идентификатор файла
            file_name: Имя файла
            text: Текст файла для обработки
            dialog_id: Идентификатор диалога, к которому относится файл
        """
        logger.info(f"Начало обработки файла: {file_name} (file_id: {file_id})")
        
        # Шаг 1: Разбиение текста на чанки
        chunks = await self._create_chunks(text, file_id, file_name, dialog_id)
        logger.info(f"Создано {len(chunks)} чанков для файла {file_name}")
        
        # Шаг 2: Векторизация чанков
        await self._vectorize_chunks(chunks)
        logger.info(f"Чанки векторизованы для файла {file_name}")
        
        # Шаг 3: Загрузка в Qdrant
        await self._load_to_qdrant(chunks)
        logger.info(f"Чанки загружены в Qdrant для файла {file_name}")
        
        # Шаг 4: Загрузка в Neo4j
        await self._load_to_neo4j(chunks)
        logger.info(f"Граф создан в Neo4j для файла {file_name}")
    
    async def _create_chunks(
        self,
        text: str,
        file_id: str,
        file_name: str,
        dialog_id: str
    ) -> List[Dict]:
        """
        Разбивает текст на чанки и добавляет метаданные.
        
        Args:
            text: Текст для разбиения
            file_id: Идентификатор файла
            file_name: Имя файла
            dialog_id: Идентификатор диалога
            
        Returns:
            Список словарей с чанками и их метаданными
        """
        logger.info(f"Разбиение текста на чанки для файла {file_name}...")
        
        # Разбиваем текст на чанки
        chunks_text = await asyncio.to_thread(
            self.chunker.make_chunks_from_text,
            text
        )
        
        # Формируем структуру чанков с метаданными
        chunks = []
        for chunk_text in chunks_text:
            chunks.append({
                "text": chunk_text,
                "file_name": file_name,
                "dialog_id": dialog_id,
                "file_id": file_id,
                "chunk_id": str(uuid.uuid4())  # Уникальный ID для каждого чанка
            })
        
        return chunks
    
    async def _vectorize_chunks(self, chunks: List[Dict]) -> None:
        """
        Векторизует все чанки, добавляя dense_vector к каждому чанку.
        
        Args:
            chunks: Список чанков для векторизации (изменяется in-place)
        """
        logger.info("Векторизация чанков...")
        
        for chunk in chunks:
            # Векторизуем текст чанка
            chunk["dense_vector"] = await asyncio.to_thread(
                self.embedder.embed,
                chunk["text"]
            )
    
    async def _load_to_qdrant(self, chunks: List[Dict]) -> None:
        """
        Загружает векторизованные чанки в Qdrant.
        
        Args:
            chunks: Список векторизованных чанков для загрузки
        """
        logger.info("Загрузка чанков в Qdrant...")
        await self.qdrant.load_in_qdrant(chunks)
    
    async def _load_to_neo4j(self, chunks: List[Dict]) -> None:
        """
        Создает граф знаний в Neo4j на основе чанков.
        
        Args:
            chunks: Список чанков для создания графа
        """
        logger.info("Создание графа в Neo4j...")
        await self.neo.create_graph(chunks)
    
    async def process_files(
        self,
        content: List[Dict],
        dialog_id: str
    ) -> None:
        """
        Обрабатывает несколько файлов для одного диалога.
        
        Args:
            content: Список файлов для обработки, каждый содержит:
                - fileId: идентификатор файла
                - fileName: имя файла
                - text: текст файла
            dialog_id: Идентификатор диалога
        """
        logger.info(f"Начало обработки {len(content)} файлов для диалога {dialog_id}")
        
        for file_data in content:
            file_id = file_data['fileId']
            file_name = file_data['fileName']
            text = file_data['text']
            
            await self.process_file(
                file_id=file_id,
                file_name=file_name,
                text=text,
                dialog_id=dialog_id
            )
        
        logger.info(f"Все файлы обработаны для диалога {dialog_id}")
