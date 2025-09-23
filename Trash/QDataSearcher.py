from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient, models as qm
import os
import dotenv

from openai import OpenAI
from Prompts import ASPECTS_SYS

dotenv.load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))


def extract_aspects_from_question(question: str) -> list[str]:
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


def dense_search(query: dict, client: QdrantClient = client, topk: int = 5) -> list[str]:
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
