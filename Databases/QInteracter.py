from qdrant_client import QdrantClient
from qdrant_client import models as qm
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    OptimizersConfigDiff, PayloadSchemaType,
    Filter, FieldCondition, MatchValue
)
from qdrant_client.http.models import PointStruct
from openai import OpenAI

from LLM.Prompts import ASPECTS_SYS

import os
import dotenv

import logging
from pathlib import Path
import sys

dotenv.load_dotenv()

logs_dir = Path("/Logs")
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "qdrant.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))


class QInteracter:
    def __init__(self, client=client):
        self.client = client
    
    async def extract_aspects_from_question(self, question: str) -> list[str]:
        """
        Extract aspects from the question
        :param question:
        :return:
        """

        logging.info("Извлечение аспектов из вопроса...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))
        resp = client.chat.completions.create(
            model=os.getenv("MS_LIGHT_MODEL"),
            messages=[
                {"role": "system", "content": ASPECTS_SYS},
                {"role": "user", "content": question}
            ]
        )
        logging.info(f"Ответ модели для извлечения аспектов: {resp.choices[0].message.content}")
        return resp.choices[0].message.content.split("||")
    
    async def dense_search(self, query: dict, topk: int = 5) -> list[str]:
        """
        Dense search. Возвращает только текст topk ближайших чанков
        :param query: вектор запроса
        :param topk:
        :return:
        """

        # фильтр. можно несколько использовать
        flt = Filter(
            must=[FieldCondition(key="dialog_id", match=MatchValue(value=query.get("dialog_id")))],
        )

        # запрос. Также тут поддерживается фукнция поиска по нескольким запросам сразу, если чуть поменять
        logging.info("Выполнение dense поиска...")

        resp = self.client.query_points(collection_name=query.get("dialog_id"), 
                                        query=query.get("dense_vector"),
                                        using="dense",
                                        query_filter=flt,
                                        limit=topk,
                                        with_payload=qm.PayloadSelectorInclude(include=["text"]),
                                        with_vectors=False)
        points = resp.points  # берем результаты первого (и единственного) запроса
        
        logging.info(f"Найдено {len(points)} ближайших чанков. Тексты чанков: {[p.payload.get('text','')[:100]+'...' for p in points]}")
        return [p.payload.get("text", "") for p in points]
    
    async def load_in_qdrant(self, chunks: list[dict]) -> None:
        """
        Функция создает коллекцию по dialog_id и загружает в нее данные. У каждого документа своя коллекция
        :param chunks: Чанки документа(вектора)
        :return:
        """
        dialog_id = chunks[0].get("dialog_id")  # у всех чанков одинаковый dialog_id

        # создать коллекцию
        # Для каждого документа своя коллекция
        logging.info(f"Создание коллекции в Qdrant для диалога {dialog_id}...")
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
        logging.info(f"Коллекция {dialog_id} создана.")

        logging.info(f"Создание индекса для метаданных в Qdrant для диалога {dialog_id}...")
        client.create_payload_index(
            collection_name=dialog_id,
            field_name="dialog_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

        logging.info(f"Индекс для метаданных в Qdrant для документа {dialog_id} создан.")


        logging.info(f"Загрузка данных в Qdrant для документа {dialog_id}...")
        # загрузка данных
        points = []
        for ch in chunks:
            points.append(
                PointStruct(
                    id=ch.get("chunk_id"),
                    vector={
                        "dense": ch.get("dense_vector"),  # list[float], len=1024
                        # "sparse": ch.get("sparse_vector")  # {indices: [...], values: [...]}
                    },
                    payload={
                        "dialog_id": dialog_id, # чанки должны хранить метаданные в себе
                        "chunk_id": ch.get("chunk_id"), # возможно, хранить его и не надо
                        "text": ch.get("text"),
                        # "title": ch.get("title"),
                        # "page": ch.get("page"),
                        # "entities": ch.get("entities"), # возможно, убрать
                        # "created_at": ch.get("created_at") # возможно, убрать
                    }
                )
            )

        """
        Пока структура чанка такая: {text: str, dense_vector: list[float],
                                    (sparse_vector: dict,) - Убираем
                                    dialog_id: str, file_id: str, file_name: str, chunk_id: str}
        После можно использовать sparse для гибридного поиска да и в целом, сделать больше фичей.
        Если захотим гибридный поиск, что bgem3 надо будет локально развернуть, HF не поддерживает sparse
        """

        logging.info(f"Загрузка {len(points)} точек в коллекцию {dialog_id}...")
        
        client.upsert(collection_name=dialog_id, points=points)

