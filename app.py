import fastapi
from fastapi.responses import JSONResponse
import json
from Embedder import embed

from QInteracter import QInteracter
from NeoInteracter import NeoInteracter

from LLMAnswer import LLM

import logging
import asyncio

logging.basicConfig(
    filename="Logs/app.log",       
    filemode="a",              
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


app = fastapi.FastAPI()
qdrant = QInteracter()
neo = NeoInteracter()
llm = LLM()

# Загрузить документ в сервис
@app.post("/load")
async def load(data: json):
    try:
        logging.info("Попытка получить данные")
        data = json.loads(data)
        logging.info("Данные успешно получены")

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return JSONResponse(content={'message': f'Не удалось сохранить файл:\n{e}'}, status_code=400)

    try:
        """
        data: {chunks: чанки list[dict[str]], doc_id: str}
        {text: str} - чанк до обработки
        """

        # векторизуем чанки и добавляем вектор, doc_id в метаданные
        logging.info("Векторизация чанков...")
        for chunk in data["chunks"]:
            chunk["dense_vector"] = asyncio.to_thread(embed, chunk["text"])
            chunk["doc_id"] = data["doc_id"]

        chunks = data["chunks"]
        logging.info("Чанки векторизованы")

        # загружаем в qdrant
        logging.info("Загрузка чанков в qdrand...")
        await qdrant.load_in_qdrant(chunks)
        logging.info("Чанки успешно загружены")

        # загружаем в neo4j
        logging.info("Загрузка чанков в neo4j")
        await neo.create_graph(chunks)
        logging.info("Чанки успешно загружены")

        return JSONResponse(content={'message':'OK'}, status_code=200)

    except Exception as e:
        logging.info(f"Ошибка: {e}")
        return JSONResponse(content={'message': f'Bad Request:\n{e}'}, status_code=500)


# Получить ответ от сервиса по запросу
@app.get("/query")
async def query(data: json):

    try:
        logging.info("Получение запроса")
        data = json.loads(data)
        """
        data: {text: str, doc_id: str, dialogue: str}
        """
        
    except Exception as e:
        logging.error("Ошибка: {e}")
        return JSONResponse(content={'message': f'Bad Request:\n{e}'}, status_code=400)

    try:
        # перефразируем запрос на основе диалога
        logging.info("Перефразирование запроса")
        question = await llm.rewrite_question_from_dialogue(data["text"], data["dialogue"])

        aspects = [] # список словарей, где каждый словарь - аспект с метаданными

        logging.info("Выделяем аспекты из запроса")
        # выделяем аспекты из запроса
        aspects_text = await qdrant.extract_aspects_from_question(question)
        for aspect in aspects_text:
            aspects.append({"text": aspect, "doc_id": data["doc_id"]})

        logging.info("Векторизуем аспекты")
        # векторизуем аспекты
        for aspect in aspects:
            aspect["dense_vector"] = asyncio.to_thread(embed, aspect["text"])

        # [{text: str, dense_vector: list, doc_id: str}, {}, ...] - aspects сейчас

        logging.info("Поиск по qdrant")
        # dense поиск по qdrant
        closest_chunks = []
        for aspect in aspects:
            chunk = await qdrant.dense_search(aspect, topk=5)
            closest_chunks.append(chunk)
        # получаем список списков, где каждый список - тексты ближайших чанков для каждого аспекта

        logging.info("Поиск по neo4j")
        # поиск по neo4j. Получаем контекст
        context = ""
        for chunks in closest_chunks:
            context += await neo.graph_context_from_chunks(chunks, data["doc_id"])

        logging.info("Запрос в LLM")
        # по контексту делаем запрос в LLM
        answer = await llm.answer_with_graph(question, context)
        logging.info("Ответ получен")

        ### Еще можно сделать формирование id ответа и потом id ответа тоже передавать

        #оборачиваем в json и возвращаем
        return JSONResponse(content={'answer': answer, 'doc_id': data["doc_id"]}, status_code=200)

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return JSONResponse(content={'message': f'Internal Server Error:\n{e}'}, status_code=500)



