"""
Генерация тестовых вопросов по диалогу и подсказкам из Neo4j (без RAG-инструментов агента).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

import json_repair
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from ..Databases.NeoInteracter import NeoInteracter
from ..LLM.Nodes.Helpers import _dialog_dicts_to_lc_messages
from ..LLM.Prompts import build_tests_generation_system_prompt

logger = logging.getLogger(__name__)


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
    """Случайные термины для подсказки модели (имя + типы узла), без технических id."""
    q = """
    MATCH (n)
    WHERE n.dialog_id = $dialog_id
    WITH labels(n) AS labels,
         coalesce(n.name, n.title, n.text, "") AS name
    WHERE name <> ""
    RETURN DISTINCT name, labels
    ORDER BY rand()
    LIMIT $n
    """
    db = graph_db.database
    with graph_db.driver.session(database=db) as session:
        rows = [dict(r) for r in session.run(q, {"dialog_id": dialog_id, "n": n})]
    return [{"name": r["name"], "labels": r.get("labels") or []} for r in rows]


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


def _parse_tests_json_payload(payload: str) -> Dict[str, Any]:
    """Строгий JSON, затем json-repair (невалидные \\ в строках от LLM)."""
    try:
        return TestsResponse.model_validate_json(payload).model_dump()
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        try:
            data = json_repair.loads(payload)
            return TestsResponse.model_validate(data).model_dump()
        except Exception as repair_e:
            logger.warning(
                "tests JSON: strict parse failed (%s), repair failed (%s)",
                e,
                repair_e,
            )
            raise e from repair_e


async def _generate_tests_payload(
    chat: ChatOpenAI,
    sys_msg: SystemMessage,
    human_msg: HumanMessage,
) -> Dict[str, Any]:
    # Groq и др.: json_schema даёт обрывы — сначала tool-calling; затем схема; потом json_object.
    for method in ("function_calling", "json_schema"):
        structured = chat.with_structured_output(TestsResponse, method=method)
        try:
            out = await structured.ainvoke([sys_msg, human_msg])
            if isinstance(out, TestsResponse):
                return out.model_dump()
            return TestsResponse.model_validate(out).model_dump()
        except Exception as e:
            logger.warning("structured output failed (method=%s): %s", method, e)

    json_chat = chat.bind(response_format={"type": "json_object"})
    raw = await json_chat.ainvoke([sys_msg, human_msg])
    content = raw.content
    text = content if isinstance(content, str) else str(content)
    try:
        payload = _extract_json_object(text)
        return _parse_tests_json_payload(payload)
    except Exception:
        logger.exception("tests generation: failed to parse model output")
        raise


async def generate_tests(
    dialog_messages: List[Dict[str, Any]],
    neo: NeoInteracter,
    dialog_id: str,
    *,
    questions_count: int,
) -> Dict[str, Any]:
    """
    Строит тест по истории диалога и случайным терминам из материала (Neo4j).
    Ровно questions_count вопросов; при несовпадении длины — один повтор запроса к LLM.
    """
    lc_messages = _dialog_dicts_to_lc_messages(dialog_messages)
    dialogue = _dialogue_text_from_lc_messages(lc_messages)

    try:
        entities = get_random_entities(neo, dialog_id, n=10)
        logger.info("Подсказки-термины для теста: %s", entities)
    except Exception:
        logger.exception("get_random_entities failed dialog_id=%s", dialog_id)
        entities = []

    entities_str = json.dumps(entities, ensure_ascii=False)

    empty_dialogue_placeholder = (
        "(Реплик чата по теме нет. Сгенерируй ровно "
        f"{questions_count} вопросов по смыслу учебных терминов ниже — проверка знаний по предмету, "
        "без вопросов про отсутствие диалога, списки или настройки.)"
    )
    if not dialogue.strip():
        dialogue_body = empty_dialogue_placeholder
    else:
        dialogue_body = dialogue

    max_completion = int(os.getenv("TESTS_GEN_MAX_COMPLETION_TOKENS", "8192"))
    chat = ChatOpenAI(
        model=os.getenv("MS_GRAPHRAG_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0.3,
        max_tokens=max_completion,
    )

    sys_content = build_tests_generation_system_prompt(questions_count)
    sys_msg = SystemMessage(content=sys_content)
    human_msg = HumanMessage(
        content=(
            f"Сгенерируй тест для обучения: ровно {questions_count} вопросов "
            "по содержанию темы (что должен знать ученик). "
            "Формат — один JSON-объект по схеме из системного сообщения.\n\n"
            f"Реплики учебного чата (из них извлекай факты о предмете, не придумывай вопросы про сам чат или формулировки запросов):\n"
            f"{dialogue_body}\n\n"
            f"Опорные термины из загруженного материала (используй для фактов по теме, не спрашивай про «список терминов», пустой он или нет): "
            f"{entities_str}"
        )
    )

    result = await _generate_tests_payload(chat, sys_msg, human_msg)
    if len(result.get("questions", [])) == questions_count:
        return result

    got = len(result.get("questions", []))
    logger.warning(
        "tests: expected %s questions, got %s — retrying once",
        questions_count,
        got,
    )
    fix_human = HumanMessage(
        content=(
            f"Предыдущий ответ содержал {got} вопросов, а нужно ровно {questions_count}. "
            "Верни заново полный один JSON-объект по схеме, с массивом \"questions\" "
            f"из ровно {questions_count} элементов. Без markdown. "
            "Вопросы — только по предмету для ученика, без мета-вопросов про чат или конфигурацию.\n\n"
            f"Реплики учебного чата:\n{dialogue_body}\n\n"
            f"Опорные термины: {entities_str}"
        )
    )
    result2 = await _generate_tests_payload(chat, sys_msg, fix_human)
    if len(result2.get("questions", [])) != questions_count:
        raise ValueError(
            f"Модель вернула {len(result2.get('questions', []))} вопросов вместо {questions_count}"
        )
    return result2
