"""
Вспомогательные функции для узлов графа агента (логирование, преобразование сообщений).
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

import logging

logger = logging.getLogger(__name__)

_AGENT_LOG_TAIL = 12


def _preview_text(text: object, limit: int = 200) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    s = s.replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _log_node_snapshot(node: str, state: Any) -> None:
    """Снимок state на входе узла: метаданные + хвост истории сообщений."""
    did = state.get("dialog_id")
    msgs = state.get("messages") or []
    n = len(msgs)
    logger.info(
        "Agent node=%s dialog_id=%s messages_total=%s input_is_safe=%s",
        node,
        did,
        n,
        state.get("input_is_safe"),
    )
    if n == 0:
        return
    start = max(0, n - _AGENT_LOG_TAIL)
    if start > 0:
        logger.info(
            "Agent node=%s dialog_id=%s messages_slice=[%s:%s] (tail %s of %s)",
            node,
            did,
            start,
            n,
            _AGENT_LOG_TAIL,
            n,
        )
    for i in range(start, n):
        m = msgs[i]
        cls = type(m).__name__
        tcalls = getattr(m, "tool_calls", None)
        if tcalls:
            logger.info(
                "Agent node=%s dialog_id=%s msg[%s] %s tool_calls=%s",
                node,
                did,
                i,
                cls,
                tcalls,
            )
            continue
        if isinstance(m, ToolMessage):
            logger.info(
                "Agent node=%s dialog_id=%s msg[%s] ToolMessage name=%r preview=%r",
                node,
                did,
                i,
                getattr(m, "name", None),
                _preview_text(getattr(m, "content", ""), 240),
            )
            continue
        body = getattr(m, "content", "")
        logger.info(
            "Agent node=%s dialog_id=%s msg[%s] %s preview=%r",
            node,
            did,
            i,
            cls,
            _preview_text(body, 240),
        )


def _make_logged_tools_node(tool_node: ToolNode):
    async def tools_node(state: Any) -> dict:
        _log_node_snapshot("tools", state)
        last = state["messages"][-1]
        tcs = getattr(last, "tool_calls", None) or []
        for tc in tcs:
            logger.info(
                "Agent tools dialog_id=%s invoking name=%r args=%r id=%r",
                state.get("dialog_id"),
                tc.get("name"),
                tc.get("args"),
                tc.get("id"),
            )
        out = await tool_node.ainvoke(state)
        for m in out.get("messages") or []:
            if isinstance(m, ToolMessage):
                logger.info(
                    "Agent tools dialog_id=%s result name=%r preview=%r",
                    state.get("dialog_id"),
                    getattr(m, "name", None),
                    _preview_text(getattr(m, "content", ""), 320),
                )
            else:
                logger.info(
                    "Agent tools dialog_id=%s extra_message type=%s preview=%r",
                    state.get("dialog_id"),
                    type(m).__name__,
                    _preview_text(getattr(m, "content", ""), 200),
                )
        return out

    return tools_node


def _dialog_dicts_to_lc_messages(dialog_messages: List[Dict]) -> list:
    """Преобразует историю из API (role/message) в сообщения LangChain."""
    out = []
    for msg in dialog_messages:
        text = msg.get("message", "")
        role = (msg.get("role") or "user").lower()
        if role == "assistant":
            out.append(AIMessage(content=text))
        else:
            out.append(HumanMessage(content=text))
    return out
