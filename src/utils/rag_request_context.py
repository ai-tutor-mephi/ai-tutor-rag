"""Контекст текущего HTTP-запроса для RAG-инструментов (LangGraph не передаёт state в @tool напрямую)."""

from contextvars import ContextVar

# True: вопрос требует обзора всего загруженного материала; rag_tool собирает контент через scroll без dense search.
is_general_document_question_cv: ContextVar[bool] = ContextVar(
    "is_general_document_question_cv", default=False
)
