import fastapi
from fastapi.responses import JSONResponse
import json
import dotenv
import os
from Handling.Embedder import Embedder

from Databases.QInteracter import QInteracter
from Databases.NeoInteracter import NeoInteracter

from LLM.LLMAnswer import LLM
from Handling.Chunker import Chunker

import asyncio
import uuid
from pydantic import BaseModel

import logging
from pathlib import Path

dotenv.load_dotenv()

base_dir = Path(__file__).resolve().parent
logs_dir = base_dir.parent / "Logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "rag.log"

logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    filemode="a", 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ContentItem(BaseModel):
    fileId: str
    fileName: str
    text: str

class LoadRequest(BaseModel):
    content: list[ContentItem]
    dialogId: str

class DialogMessage(BaseModel):
    message: str
    role: str

class QueryRequest(BaseModel):
    dialogId: str
    dialogMessages: list[DialogMessage]
    question: str

rag = fastapi.FastAPI()

qdrant: QInteracter = None
neo: NeoInteracter = None
llm: LLM = None
embedder: Embedder = None
chunker: Chunker = None

@rag.on_event("startup")
async def startup_event():
    global qdrant, neo, llm, embedder, chunker
    qdrant = QInteracter()
    neo = NeoInteracter()
    llm = LLM()
    embedder = Embedder()
    chunker = Chunker()

# Загрузить документ в сервис
@rag.post("/load")
async def load(data: LoadRequest):
    try:
        logging.info("Попытка получить данные")
        data = data.dict()
        logging.info("Данные успешно получены")

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return JSONResponse(content={'message': f'Не удалось сохранить файл:\n{e}'}, status_code=400)

    try:
        """
        {
          "content": [
                      {
                      "fileId": "идентификатор_документа", - ЭТО МОЙ doc_id
                      "fileName": "...", 
                      "text": "..."
                      }
          ],
          "dialogId": "идентификатор_диалога"
        }

        """
        content = data['content']
        dialog_id = data['dialogId']
        for i in range(len(content)):
            file_id = data['content'][i]['fileId']
            file_name = data['content'][i]['fileName']
            text = data['content'][i]['text']


            # ДОБАВИТЬ РАЗБИЕНИЕ ТЕКСТА НА ЧАНКИ!!!

            # разбиваем текст на чанки
            chunks = []
            logging.info("Разбиваем текст на чанки...")
            chunks_text = await asyncio.to_thread(chunker.make_chunks_from_text, text)
            for chunk in chunks_text:
                chunks.append({"text": chunk,
                               "file_name": file_name,
                               "dialog_id": dialog_id,
                               "file_id": file_id,
                               "chunk_id": str(uuid.uuid4())})

            logging.info("Чанки успешно получены")

            # векторизуем чанки и добавляем вектор, dialog_id в метаданные добавили выше
            logging.info("Векторизация чанков...")
            for chunk in chunks:
                chunk["dense_vector"] = await asyncio.to_thread(embedder.embed, chunk["text"])


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
@rag.post("/query")
async def query(data: QueryRequest):

    try:
        logging.info("Получение запроса")
        data = data.dict()
        """
          {
            "dialogId": "идентификатор_диалога",
            "dialogMessages": [
                    {
                    "message": "...", 
                    "role": "..."
                    },
                    ...
            ],
            "question": "..."
            }
        """
        dialog_id = data["dialogId"]
        dialog_messages = data["dialogMessages"]
        question = data["question"]
        
    except Exception as e:
        logging.error("Ошибка: {e}")
        return JSONResponse(content={'message': f'Bad Request:\n{e}'}, status_code=400)

    try:
        # перефразируем запрос на основе диалога
        logging.info("Перефразирование запроса")
        question = await llm.rewrite_question_from_dialogue(question=question, dialogue=''.join([f"{msg['role']}: {msg['message']}" for msg in dialog_messages]))

        aspects = []  # список словарей, где каждый словарь - аспект с метаданными

        logging.info("Выделяем аспекты из запроса")
        # выделяем аспекты из запроса
        aspects_text = await qdrant.extract_aspects_from_question(question)
        for aspect in aspects_text:
            aspects.append({"text": aspect, "dialogId": dialog_id})

        logging.info("Векторизуем аспекты")
        # векторизуем аспекты
        for aspect in aspects:
            aspect["dense_vector"] = await asyncio.to_thread(embedder.embed, aspect["text"])

        # [{text: str, dense_vector: list, dialog_id: str}, {}, ...] - aspects сейчас

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
            ctx = await neo.graph_context_from_chunks(chunks, dialog_id=dialog_id)
            context += ctx.get("context_text", "")

        logging.info("Запрос в LLM")
        # по контексту делаем запрос в LLM
        answer = await llm.answer_with_graph(question, context)
        logging.info("Ответ получен")

        ### Еще можно сделать формирование id ответа и потом id ответа тоже передавать

        #оборачиваем в json и возвращаем
        return JSONResponse(content={'answer': answer, 'dialogId': dialog_id}, status_code=200)

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return JSONResponse(content={'message': f'Internal Server Error:\n{e}'}, status_code=500)



