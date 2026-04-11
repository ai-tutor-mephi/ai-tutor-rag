"""
Генерация тестовых вопросов по диалогу и подсказкам из Neo4j (без RAG-инструментов агента).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from Databases.NeoInteracter import NeoInteracter
from LLM.Nodes.Helpers import _dialog_dicts_to_lc_messages
from LLM.Prompts import TESTS_GENERATION_SYS
from utils.MyLogs import setup_logger

logger = setup_logger(__file__)


class Question(BaseModel):
    question: str
    variants: List[str]
    gold_answer: str

    @field_validator("variants")
    @classmethod
    def exactly_four_variants(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("variants must contain exactly 4 strings")
        return v

    @model_validator(mode="after")
    def gold_must_be_a_variant(self) -> Question:
        if self.gold_answer not in self.variants:
            raise ValueError("gold_answer must be identical to one of variants")
        return self


class TestsResponse(BaseModel):
    test_name: str
    questions: List[Question]


def get_random_entities(graph_db: NeoInteracter, dialog_id: str, n: int = 10) -> List[Dict[str, Any]]:
    """Случайные узлы графа с данным dialog_id (имя / подписи для подсказки модели)."""
    q = """
    MATCH (n)
    WHERE n.dialog_id = $dialog_id
    RETURN elementId(n) AS id,
           labels(n) AS labels,
           coalesce(n.name, n.title, n.text, "") AS name
    ORDER BY rand()
    LIMIT $n
    """
    db = graph_db.database
    with graph_db.driver.session(database=db) as session:
        return [dict(r) for r in session.run(q, {"dialog_id": dialog_id, "n": n})]


def _dialogue_text_from_lc_messages(messages: list) -> str:
    lines: List[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"User: {c}")
        elif isinstance(m, AIMessage):
            c = m.content if isinstance(m.content, str) else str(m.content or "")
            if c.strip():
                lines.append(f"Assistant: {c}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> str:
    """Вырезает первый JSON-объект из ответа (на случай лишнего текста или ```json)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return t[start : end + 1]


async def generate_tests(
    dialog_messages: List[Dict[str, Any]],
    neo: NeoInteracter,
    dialog_id: str,
) -> Dict[str, Any]:
    """
    Строит тест по истории диалога (и случайным сущностям графа). Возвращает dict для JSON-ответа API.
    """
    lc_messages = _dialog_dicts_to_lc_messages(dialog_messages)
    dialogue = _dialogue_text_from_lc_messages(lc_messages)

    try:
        entities = get_random_entities(neo, dialog_id, n=10)
    except Exception:
        logger.exception("get_random_entities failed dialog_id=%s", dialog_id)
        entities = []

    entities_str = json.dumps(entities, ensure_ascii=False)

    if not dialogue.strip():
        dialogue = (
            "(Реплик в истории нет. Сгенерируй вопросы по сущностям из графа, без выдумывания фактов вне них.)"
        )

    chat = ChatOpenAI(
        model=os.getenv("MS_GRAPHRAG_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0.3,
    )

    sys_msg = SystemMessage(content=TESTS_GENERATION_SYS)
    human_msg = HumanMessage(
        content=(
            "Ниже транскрипт диалога. Сгенерируй тест строго в одном JSON-объекте по схеме из системного сообщения.\n\n"
            f"{dialogue}\n\n"
            f"Подсказка — случайные узлы графа для этого dialog_id: {entities_str}"
        )
    )

    structured = chat.with_structured_output(TestsResponse)
    try:
        out = await structured.ainvoke([sys_msg, human_msg])
        if isinstance(out, TestsResponse):
            return out.model_dump()
        return TestsResponse.model_validate(out).model_dump()
    except Exception as e:
        logging.warning("structured output failed (%s), falling back to raw JSON parse", e)

    raw = await chat.ainvoke([sys_msg, human_msg])
    content = raw.content
    text = content if isinstance(content, str) else str(content)
    try:
        payload = _extract_json_object(text)
        parsed = TestsResponse.model_validate_json(payload)
        return parsed.model_dump()
    except (ValidationError, ValueError, json.JSONDecodeError):
        logger.exception("tests generation: failed to parse model output")
        raise
