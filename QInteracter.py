from qdrant_client import QdrantClient, models as qm
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    OptimizersConfigDiff, PayloadSchemaType,
    Filter, FieldCondition, MatchValue
)
from qdrant_client.http.models import PointStruct
from openai import OpenAI

from Prompts import ASPECTS_SYS

import os
import dotenv


dotenv.load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))


class QInteracter:
    def __init__(self, client=client):
        self.client = client
    
    def extract_aspects_from_question(self, question: str) -> list[str]:
        """
        Extract aspects from the question
        :param question:
        :return:
        """
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))
        resp = client.chat.completions.create(
            model=os.getenv("MS_LIGHT_MODEL"),
            messages=[
                {"role": "system", "content": ASPECTS_SYS},
                {"role": "user", "content": question}
            ]
        )
        return resp.choices[0].message.content.split("||")
    
    def dense_search(self, query: dict, topk: int = 5) -> list[str]:
        """
        Dense search. Возвращает только текст topk ближайших чанков
        :param query: вектор запроса
        :param client:
        :param topk:
        :return:
        """

        # фильтр. можно несколько использовать
        # flt = Filter(
        #     must=[FieldCondition(key="doc_id", match=MatchValue(value=query.get("doc_id")))],
        # )

        # запрос. Также тут поддерживается фукнция поиска по нескольким запросам сразу, если чуть поменять
        q = qm.Query(
            query_vector=("dense", query.get("dense_vector")),
            # query_filter=flt,
            limit=topk,
            with_vectors=False,
            # Можно запросить только нужные поля payload:
            with_payload=qm.PayloadSelectorInclude(include=["text"])
        )

        resp = client.query_points(collection_name=query.get("doc_id"), query=[q])
        points = resp[0].points  # берем результаты первого (и единственного) запроса
        return [p.payload.get("text", "") for p in points]
    
    def load_in_qdrant(self, chunks: list[dict]) -> None:
        """
        Функция создает коллекцию по id документа и загружает в нее данные. У каждого документа своя коллекция
        :param chunks: Чанки документа(вектора)
        :param client: QdrantClient
        :return:
        """
        doc_id = chunks[0].get("doc_id")

        # создать коллекцию
        # Для каждого документа своя коллекция
        client.create_collection(
            collection_name=doc_id,  # коллекцию называем по идентификатору документа, возможно, потом по другому
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
                "sparse": SparseVectorParams()
            },
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2
            ),
        )

        client.create_payload_index(
            collection_name=doc_id,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD
        )


        # загрузка данных
        points = []
        for ch in chunks:
            points.append(
                PointStruct(
                    id=ch.get("chunk_id"),
                    vector={
                        "dense": ch.get("dense_vector"),  # list[float], len=1024
                        "sparse": ch.get("sparse_vector")  # {indices: [...], values: [...]}
                    },
                    payload={
                        "doc_id": doc_id, # чанки должны хранить метаданные в себе
                        # "chunk_id": ch.get("chunk_id"), # возможно, хранить его и не надо
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
                                    doc_id: str}
        После можно использовать sparse для гибридного поиска да и в целом, сделать больше фичей.
        Если захотим гибридный поиск, что bgem3 надо будет локально развернуть, HF не поддерживает sparse
        """

        client.upsert(collection_name=doc_id, points=points)

