from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    OptimizersConfigDiff, PayloadSchemaType
)
from qdrant_client.http.models import PointStruct
import os
import dotenv

# def extract_data_from_json(data: json) -> list:
# эту фукнцию нужно будет на уровень выше переместить, до извлечения чанков


dotenv.load_dotenv()

client =QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))


def load_in_qdrant(chunks: list[dict], client: QdrantClient=client) -> None:
    """
    Функция создает коллекцию по id документа и загружает в нее данные. У каждого документа своя коллекция
    :param chunks: Чанки документа(вектора)
    :param client: QdrantClient
    :return:
    """
    doc_id = chunks[0].get("doc_id")

    # создать коллекцию
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
    # определить у нас для каждого документа своя коллекция или у нас одна коллекция в qdrant
    # решили, что разные коллекции

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


