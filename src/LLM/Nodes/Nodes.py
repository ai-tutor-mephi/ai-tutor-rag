"""
Узлы и маршрутизация графа LangGraph для агента.
"""

from __future__ import annotations

import os
import requests
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict

from .Helpers import _log_node_snapshot, _preview_text
from utils.MyLogs import setup_logger

load_dotenv()

logger = setup_logger(__file__)


class AgentState(TypedDict):
    """Состояние агента для LangGraph."""

    messages: Annotated[list, add_messages]
    dialog_id: str
    input_is_safe: bool


def reject_input_node(state: AgentState) -> AgentState:
    """Узел для отклонения небезопасных входных сообщений."""
    _log_node_snapshot("reject_input", state)
    messages = state["messages"]
    logger.info(
        "reject_input: dialog_id=%s rejecting unsafe input",
        state.get("dialog_id"),
    )

    messages.append(
        AIMessage(
            content=(
                "Запрос выглядит небезопасным или похожим на попытку обойти ограничения. "
                "Переформулируй его обычным способом: просто опиши, что именно тебе нужно."
            )
        )
    )

    return {
        **state,
        "messages": messages,
    }


def input_guard_node(state: AgentState) -> dict:
    """Проверяет входное сообщение пользователя и сохраняет результат в state."""
    _log_node_snapshot("input_guard", state)
    messages = state["messages"]
    last_message = messages[-1]

    user_text = last_message.content if hasattr(last_message, "content") else str(last_message)
    logger.info(
        "input_guard: dialog_id=%s validating user_text preview=%r",
        state.get("dialog_id"),
        _preview_text(user_text, 300),
    )

    server_url = os.getenv("GUARDRAILS_SERVER_URL")

    if not server_url:
        logger.debug(
            "input_guard_node: GUARDRAILS_SERVER_URL не задан => allow (dialog_id=%s)",
            state.get("dialog_id"),
        )
        return {"input_is_safe": True}

    try:
        validate_url = f"{server_url.rstrip('/')}/guards/input_guard/validate"
        timeout_s = float(os.getenv("GUARDRAILS_VALIDATE_TIMEOUT", "120"))
        logger.debug("input_guard_node: POST %s", validate_url)
        response = requests.post(
            validate_url,
            json={"llmOutput": user_text},
            timeout=timeout_s,
        )

        if response.status_code != 200:
            logger.warning(
                "input_guard_node: non-200 response from Guardrails (status=%s, dialog_id=%s, guard_url=%s)",
                response.status_code,
                state.get("dialog_id"),
                validate_url,
            )

            return {"input_is_safe": True}

        try:
            data = response.json()
        except Exception:
            logger.exception(
                "input_guard_node: failed to parse JSON (dialog_id=%s, url=%s)",
                state.get("dialog_id"),
                validate_url,
            )
            raise

        is_passed = data.get("validationPassed", False)

        logger.debug(
            "input_guard_node: validationPassed=%s (dialog_id=%s)",
            is_passed,
            state.get("dialog_id"),
        )
        return {"input_is_safe": is_passed}

    except requests.RequestException:
        logger.exception(
            "input_guard_node: requests exception (dialog_id=%s, server_url=%s)",
            state.get("dialog_id"),
            server_url,
        )
        return {"input_is_safe": True}


def route_after_guard(state: AgentState) -> Literal["agent", "reject_input"]:
    """После guard: основной агент или отказ."""
    is_safe = state.get("input_is_safe", True)
    nxt = "agent" if is_safe else "reject_input"
    logger.info(
        "route_after_guard dialog_id=%s -> %s (input_is_safe=%s)",
        state.get("dialog_id"),
        nxt,
        is_safe,
    )
    return nxt


def create_agent_node(llm: ChatOpenAI, tools: list):
    """Узел агента: LLM с инструментами."""
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState):
        _log_node_snapshot("agent", state)
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        logger.info(
            "agent LLM: dialog_id=%s response_type=%s tool_calls=%s",
            state.get("dialog_id"),
            type(response).__name__,
            getattr(response, "tool_calls", None),
        )
        return {"messages": [response]}

    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Ветвление после ответа агента: инструменты или конец."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(
            "should_continue dialog_id=%s -> tools (n_calls=%s)",
            state.get("dialog_id"),
            len(last_message.tool_calls),
        )
        return "tools"

    logger.info("should_continue dialog_id=%s -> end", state.get("dialog_id"))
    return "end"
