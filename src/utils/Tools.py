from pydantic import BaseModel
from typing import List

import dotenv
import os
import re

import asyncio
from langchain_core.tools import tool

import logging

from ..Databases.QInteracter import QInteracter
from ..Databases.NeoInteracter import NeoInteracter
from ..Handling.Embedder import Embedder
from .rag_request_context import is_general_document_question_cv


dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Вопросы «про весь документ» плохо матчятся по одному вектору «содержание файла» — добавляем широкие поисковые формулировки.
_DOC_SCOPE_MARKERS = (
    "конспект",
    "документ",
    "файл",
    "загружен",
    "содержан",
    "перескаж",
    "кратк",
    "суть",
    "о чём",
    "о чем",
    "что в ",
    "что говор",
    "что сказа",
    "изложи",
    "резюме",
    "summary",
    "summarize",
    "tl;dr",
    "gist",
    "uploaded",
    "document",
    "the file",
    "what's in",
    "what is in",
    "what does",
)


def _normalize_query_key(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _is_document_scope_question(question: str) -> bool:
    if not question or not question.strip():
        return False
    low = question.lower()
    return any(m in low for m in _DOC_SCOPE_MARKERS)


def _dedupe_chunks(chunks: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in chunks:
        s = (c or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _retrieval_queries(question: str, aspects: List[str]) -> List[str]:
    """Исходный вопрос + аспекты + (для обзорных запросов) широкие тематические строки для dense search."""
    seen: set[str] = set()
    ordered: List[str] = []

    def add(q: str) -> None:
        q = (q or "").strip()
        if not q:
            return
        k = _normalize_query_key(q)
        if k in seen:
            return
        seen.add(k)
        ordered.append(q)

    add(question)
    for a in aspects:
        add(a)
    if _is_document_scope_question(question):
        for extra in (
            "основные темы разделы и ключевые идеи учебного материала",
            "определения термины факты и выводы из текста документа",
        ):
            add(extra)
    return ordered


class DialogMessage(BaseModel):
    message: str
    role: str

class QueryRequest(BaseModel):
    dialogId: str
    dialogMessages: List[DialogMessage]
    question: str

_embedder = Embedder()
_qdrant = QInteracter()
_neo = NeoInteracter()


@tool
async def rag_tool(question: str, dialog_id: str) -> str:
    """
    Инструмент для получения контекста из RAG системы.
    
    Извлекает ключевые аспекты из вопроса, находит релевантные чанки
    в векторной БД и строит графовый контекст для ответа.
    
    
    Args:
        question: Вопрос пользователя, на который нужно найти контекст
        dialog_id: Идентификатор диалога (указан в системном сообщении как "Current dialog ID")
        
    Returns:
        Контекст в виде текста для генерации ответа
    """
    is_general = is_general_document_question_cv.get()

    if is_general:
        logger.info("Режим общего вопроса: scroll всех чанков коллекции (без dense search и без Neo4j)")
        chunks = await _qdrant.scroll_all_chunk_texts(dialog_id)
        if not chunks:
            return ""

        max_chars = int(os.getenv("FULL_DOC_CONTEXT_MAX_CHARS", "120000"))
        assembled = "\n\n---\n\n".join(chunks)
        if len(assembled) > max_chars:
            assembled = (
                assembled[:max_chars]
                + "\n\n[... текст обрезан по FULL_DOC_CONTEXT_MAX_CHARS; для полноты ответа опирайся на приведённое ...]"
            )

        return (
            "[Контекст: полный набор фрагментов документов диалога, собранных из всех чанков без семантического отбора]\n"
            + assembled
        )

    logger.info("Узкий вопрос: аспекты + dense search + граф")
    logger.info("Выделяем аспекты из запроса")
    aspects_text = await _qdrant.extract_aspects_from_question(question)
    queries = _retrieval_queries(question, aspects_text)
    topk = 10 if _is_document_scope_question(question) else 5

    aspects = []
    for qtext in queries:
        vec = await asyncio.to_thread(_embedder.embed, qtext)
        aspects.append(
            {"text": qtext, "dialog_id": dialog_id, "dense_vector": vec}
        )

    logger.info("Поиск по qdrant (запросов: %s, topk=%s)", len(aspects), topk)
    all_chunks: List[str] = []
    for aspect in aspects:
        chunks = await _qdrant.dense_search(aspect, topk=topk)
        all_chunks.extend(chunks)

    all_chunks = _dedupe_chunks(all_chunks)

    logger.info("Поиск по neo4j")
    if all_chunks:
        graph_data = await _neo.graph_context_from_chunks(all_chunks, dialog_id=dialog_id)
        context = graph_data.get("context_text", "")
    else:
        context = ""

    return context

tools = [rag_tool]
tools_by_name = {tool.name: tool for tool in tools}