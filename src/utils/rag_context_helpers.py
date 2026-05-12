"""
Общая логика: сколько и какие тексты чанков из Qdrant попадают в контекст для LLM.

Используется в rag_tool (узкий RAG) и в QueryService._build_graph_context (наследие пайплайна).
"""

from __future__ import annotations

import os
from typing import List


def dedupe_chunks(chunks: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in chunks:
        s = (c or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def dense_context_chunk_limit() -> int:
    """Сколько чанков отдавать в текст контекста для LLM (после dedupe, по порядку релевантности)."""
    n = int(os.getenv("RAG_DENSE_CONTEXT_MAX_CHUNKS", "4"))
    return max(1, min(n, 50))


def dense_chunks_context_section(chunks: List[str]) -> str:
    """
    Собирает один блок с текстами чанков для подстановки в промпт.
    Список chunks уже должен быть усечён до нужного топа (см. dense_context_chunk_limit).
    """
    if not chunks:
        return ""
    max_chars = int(os.getenv("RAG_DENSE_CHUNKS_MAX_CHARS", "80000"))
    assembled = "\n\n---\n\n".join(chunks)
    if len(assembled) > max_chars:
        assembled = (
            assembled[:max_chars]
            + "\n\n[... фрагменты обрезаны по RAG_DENSE_CHUNKS_MAX_CHARS ...]"
        )
    return (
        "[Фрагменты документов (релевантные отрывки из загруженных материалов)]\n"
        + assembled
    )


def graph_context_header() -> str:
    return "[Граф знаний (связи и краткие описания сущностей из ваших документов)]\n"
