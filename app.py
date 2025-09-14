import fastapi
import json
from Embedder import embed
from QDataLoader import load_in_qdrant
from NeoDataLoader import create_graph
from QDataSearcher import extract_aspects_from_question, dense_search
from NeoDataSearcher import graph_context_from_chunks
from LLMAnswer import answer_with_graph

app = fastapi.FastAPI()

# Загрузить документ в сервис
@app.post("/load")
def load(data: json):
    data = json.loads(data)
    """
    data: {chunks: чанки list[dict[str]], doc_id: str}
    {text: str} - чанк до обработки
    """

    # векторизуем чанки и добавляем вектор, doc_id в метаданные
    for chunk in data["chunks"]:
        chunk["dense_vector"] = embed(chunk["text"])
        chunk["doc_id"] = data["doc_id"]

    chunks = data["chunks"]

    # загружаем в qdrant
    load_in_qdrant(chunks)

    # загружаем в neo4j
    create_graph(chunks)

    return {"status": "200"}




# Получить ответ от сервиса по запросу
@app.get("/query")
def query(data: json):
    data = json.loads(data)
    """
    data: {text: str, doc_id: str}
    """

    aspects = [] # список словарей, где каждый словарь - аспект с метаданными

    # выделяем аспекты из запроса
    aspects_text = extract_aspects_from_question(data["text"])
    for aspect in aspects_text:
        aspects.append({"text": aspect, "doc_id": data["doc_id"]})

    # векторизуем аспекты
    for aspect in aspects:
        aspect["dense_vector"] = embed(aspect["text"])

    # [{text: str, dense_vector: list, doc_id: str}, {}, ...] - aspects сейчас


    # dense поиск по qdrant
    closest_chunks = []
    for aspect in aspects:
        closest_chunks.append(dense_search(aspect, topk=5))

    # получаем список списков, где каждый список - тексты ближайших чанков для каждого аспекта

    # поиск по neo4j. Получаем контекст
    context = ""
    for chunks in closest_chunks:
        context += graph_context_from_chunks(chunks, data["doc_id"])

    # по контексту делаем запрос в LLM
    answer = answer_with_graph(data["text"], context)

    ### Еще можно сделать формирование id ответа и потом id ответа тоже передавать

    #оборачиваем в json и возвращаем
    return json.dumps({"answer": answer, "doc_id": data["doc_id"]})



