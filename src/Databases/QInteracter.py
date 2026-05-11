"""
Модуль для работы с векторной базой данных Qdrant.

Этот модуль предоставляет интерфейс для:
- Загрузки векторизованных чанков в Qdrant
- Семантического поиска по векторной БД
- Извлечения аспектов из вопросов через LLM

Qdrant используется для хранения векторизованных чанков документов
и быстрого поиска релевантных фрагментов по запросу пользователя.
"""

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    OptimizersConfigDiff, PayloadSchemaType,
    Filter, FieldCondition, MatchValue
)
from qdrant_client.http.models import PointStruct
from openai import OpenAI

from ..LLM.Prompts import ASPECTS_SYS

import os
import dotenv

import logging

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Инициализация клиента Qdrant
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))


class QInteracter:
    """
    Класс для взаимодействия с векторной БД Qdrant.
    
    Предоставляет методы для:
    - Загрузки векторизованных чанков (создание коллекций по dialog_id)
    - Семантического поиска по векторной БД
    - Извлечения ключевых аспектов из вопросов
    """
    def __init__(self, client=client):
        self.client = client
    
    async def extract_aspects_from_question(self, question: str) -> list[str]:
        """
        Извлекает ключевые аспекты из вопроса пользователя.
        
        Использует LLM для выделения основных тем и концепций,
        которые нужно найти в базе знаний для ответа на вопрос.
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Список аспектов, разделенных символом "||"
        """

        logger.info("Извлечение аспектов из вопроса...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"))
        resp = client.chat.completions.create(
            model=os.getenv("MS_LIGHT_MODEL"),
            messages=[
                {"role": "system", "content": ASPECTS_SYS},
                {"role": "user", "content": question}
            ]
        )
        raw = resp.choices[0].message.content or ""
        logger.info(f"Ответ модели для извлечения аспектов: {raw}")
        return [a.strip() for a in raw.split("||") if a.strip()]
    
    async def dense_search(self, query: dict, topk: int = 5) -> list[str]:
        """
        Выполняет семантический поиск в векторной БД.
        
        Ищет top-k наиболее похожих чанков по векторному представлению запроса.
        Поиск ограничен коллекцией, соответствующей dialog_id.
        
        Args:
            query: Словарь с вектором запроса и dialog_id:
                - dense_vector: векторное представление запроса
                - dialog_id: идентификатор диалога (определяет коллекцию)
            topk: Количество ближайших чанков для возврата
            
        Returns:
            Список текстов найденных чанков
        """

        # фильтр. можно несколько использовать
        flt = Filter(
            must=[FieldCondition(key="dialog_id", match=MatchValue(value=query.get("dialog_id")))],
        )

        # запрос. Также тут поддерживается фукнция поиска по нескольким запросам сразу, если чуть поменять
        logger.info("Выполнение dense поиска...")

        resp = self.client.query_points(collection_name=query.get("dialog_id"), 
                                        query=query.get("dense_vector"),
                                        using="dense",
                                        query_filter=flt,
                                        limit=topk,
                                        with_payload=qm.PayloadSelectorInclude(include=["text"]),
                                        with_vectors=False)
        points = resp.points  # берем результаты первого (и единственного) запроса
        
        logger.info(f"Найдено {len(points)} ближайших чанков. Тексты чанков: {[p.payload.get('text','')[:100]+'...' for p in points]}")
        return [p.payload.get("text", "") for p in points]

    async def scroll_all_chunk_texts(self, dialog_id: str) -> list[str]:
        """
        Возвращает тексты всех точек коллекции диалога, отсортированные по файлу и chunk_id.

        Используется для «общих» вопросов: без эмбеддингов и без top-k отбора.
        """
        if not self.client.collection_exists(collection_name=dialog_id):
            logger.info("Коллекция %s не существует — scroll пустой", dialog_id)
            return []

        rows: list[dict] = []
        next_offset = None
        while True:
            scroll_kwargs: dict = {
                "collection_name": dialog_id,
                "limit": 256,
                "with_payload": qm.PayloadSelectorInclude(
                    include=["text", "file_name", "chunk_id"]
                ),
                "with_vectors": False,
            }
            if next_offset is not None:
                scroll_kwargs["offset"] = next_offset

            points, next_offset = self.client.scroll(**scroll_kwargs)

            for p in points:
                pl = p.payload or {}
                chunk_id_val = pl.get("chunk_id")
                if chunk_id_val is None:
                    chunk_id_val = str(p.id)
                text = (pl.get("text") or "").strip()
                if not text:
                    continue
                rows.append(
                    {
                        "text": text,
                        "file_name": pl.get("file_name") or "",
                        "chunk_id": str(chunk_id_val),
                    }
                )

            if next_offset is None:
                break

        rows.sort(key=lambda r: (r["file_name"], r["chunk_id"]))
        seen: set[str] = set()
        ordered: list[str] = []
        for r in rows:
            t = r["text"]
            if t in seen:
                continue
            seen.add(t)
            ordered.append(t)
        logger.info("scroll_all_chunk_texts: dialog_id=%s chunks=%s", dialog_id, len(ordered))
        return ordered
    
    async def load_in_qdrant(self, chunks: list[dict]) -> None:
        """
        Загружает векторизованные чанки в Qdrant.
        
        Для каждого диалога создается отдельная коллекция (если не существует).
        Если коллекция уже существует, новые чанки добавляются к существующим.
        
        Структура чанка:
        {
            "text": str,
            "dense_vector": list[float],
            "dialog_id": str,
            "file_id": str,
            "file_name": str,
            "chunk_id": str
        }
        
        Args:
            chunks: Список векторизованных чанков для загрузки
        """
        """
        Пока структура чанка такая: {text: str, dense_vector: list[float],
                                    (sparse_vector: dict,) - пока нет
                                    dialog_id: str, file_id: str, file_name: str, chunk_id: str}
        После можно использовать sparse для гибридного поиска да и в целом, сделать больше фичей.
        """

        dialog_id = chunks[0].get("dialog_id")  # у всех чанков одинаковый dialog_id
        file_id = chunks[0].get("file_id")


        
        points = []
        for ch in chunks:
            points.append(
                PointStruct(
                    id=ch.get("chunk_id", ""),
                    vector={
                        "dense": ch.get("dense_vector"),  # list[float], len=1024
                        # "sparse": ch.get("sparse_vector", "")   # {indices: [...], values: [...]}
                    },
                    payload={
                        "dialog_id": dialog_id,
                        "chunk_id": ch.get("chunk_id"), # возможно, хранить его и не надо
                        "text": ch.get("text"),
                        "file_id": file_id,
                        # "title": ch.get("title"),
                        # "page": ch.get("page"),
                        # "entities": ch.get("entities"), # возможно, убрать
                        # "created_at": ch.get("created_at") # возможно, убрать
                    }
                )
            )
        
        
        if client.collection_exists(collection_name=dialog_id):
            logger.info(f"Коллекция {dialog_id} уже существует. Добавление новых данных...")
            client.upsert(collection_name=dialog_id, points=points)
            return


        else:
            # создать коллекцию
            # Для каждого диалога своя коллекция
            logger.info(f"Создание коллекции в Qdrant для диалога {dialog_id}...")
            client.create_collection(
                collection_name=dialog_id,  # коллекцию называем по идентификатору диалога, возможно, потом по другому
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE),
                    "sparse": VectorParams(size=10000, distance=Distance.DOT) # пример для sparse
                },
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2
                ),
            )
            logger.info(f"Коллекция {dialog_id} создана.")

            logger.info(f"Создание индекса для метаданных в Qdrant для диалога {dialog_id}...")
            client.create_payload_index(
                collection_name=dialog_id,
                field_name="dialog_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Загрузка {len(points)} точек в коллекцию {dialog_id}...")
            client.upsert(collection_name=dialog_id, points=points)
            return

