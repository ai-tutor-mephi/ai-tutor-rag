"""
Модуль для агента на основе LangGraph
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .LLMAnswer import LLM
from .Nodes.Helpers import _dialog_dicts_to_lc_messages, _make_logged_tools_node
from .Nodes.Nodes import (
    AgentState,
    create_agent_node,
    input_guard_node,
    reject_input_node,
    route_after_guard,
    should_continue,
)
from .Prompts import AGENT_CONTEXT_SYS
from ..utils.Tools import tools
from ..utils.rag_request_context import is_general_document_question_cv

load_dotenv()

logger = logging.getLogger(__name__)


def _message_role(msg: BaseMessage) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    return getattr(msg, "type", msg.__class__.__name__)


def _log_agent_context(
    dialog_id: str,
    system_message: SystemMessage,
    history_messages: Sequence[BaseMessage],
    user_message: HumanMessage,
) -> None:
    """
    Пишет в лог то, что реально уходит в граф:
    - полный system prompt;
    - краткую сводку по истории;
    - при включённом LOG_AGENT_CONTEXT=1 — полные тексты сообщений.
    """
    try:
        log_full = os.getenv("LOG_AGENT_CONTEXT", "0") == "1"

        logger.info(
            "Agent context (dialog_id=%s): system_len=%s history_msgs=%s user_len=%s",
            dialog_id,
            len(str(system_message.content)) if system_message.content is not None else 0,
            len(history_messages),
            len(str(user_message.content)) if user_message.content is not None else 0,
        )

        logger.info(
            "Agent context system (dialog_id=%s): %s",
            dialog_id,
            system_message.content,
        )

        if history_messages:
            logger.info("Agent context history summary (dialog_id=%s):", dialog_id)
            for idx, m in enumerate(history_messages):
                content_str = str(getattr(m, "content", ""))
                preview = content_str[:200].replace("\n", " ")
                logger.info(
                    "  [%s] role=%s len=%s preview=%s",
                    idx,
                    _message_role(m),
                    len(content_str),
                    preview,
                )

        if log_full:
            logger.info("Agent context FULL history (dialog_id=%s):", dialog_id)
            for idx, m in enumerate(history_messages):
                logger.info(
                    "  FULL[%s] role=%s content=%r",
                    idx,
                    _message_role(m),
                    getattr(m, "content", None),
                )

            logger.info(
                "Agent context FULL user (dialog_id=%s): %r",
                dialog_id,
                user_message.content,
            )
    except Exception:
        logger.exception("Failed to log agent context (dialog_id=%s)", dialog_id)


class Agent:
    """Агент для обработки запросов через LangGraph с использованием инструментов."""

    def __init__(self, llm: Optional[LLM] = None):
        self.chat_llm = ChatOpenAI(
            model=os.getenv("MS_GRAPHRAG_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            temperature=0,
        )

        self.agent_node = create_agent_node(self.chat_llm, tools)
        self.tool_node = ToolNode(tools)
        self.tools_node_logged = _make_logged_tools_node(self.tool_node)

        self.graph = self.init_graph()

    def init_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", self.tools_node_logged)
        graph.add_node("input_guard", input_guard_node)
        graph.add_node("reject_input", reject_input_node)
        logger.info(
            "Agent graph initialized (no cross-request checkpoint; history только из тела запроса)"
        )

        graph.add_edge(START, "input_guard")

        graph.add_conditional_edges(
            "input_guard",
            route_after_guard,
            {
                "agent": "agent",
                "reject_input": "reject_input",
            },
        )

        graph.add_edge("reject_input", END)

        graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )

        graph.add_edge("tools", "agent")

        return graph.compile()

    async def run(
        self,
        question: str,
        dialog_id: str,
        dialog_messages: List[Dict],
        *,
        is_general_document_question: bool = False,
    ) -> str:
        logger.info(
            "Agent.run start dialog_id=%s question_len=%s history_turns=%s",
            dialog_id,
            len(question) if question is not None else None,
            len(dialog_messages) if dialog_messages else 0,
        )
        # Дополнительное логирование контекста агента можно включить через LOG_AGENT_CONTEXT=1|true.
        log_agent_ctx = os.getenv("LOG_AGENT_CONTEXT", "").lower() in ("1", "true", "yes")

        system_content = AGENT_CONTEXT_SYS + f"""

The rag_tool requires the dialog_id parameter. Use this dialog ID: {dialog_id}
"""

        system_message = SystemMessage(content=system_content)
        user_message = HumanMessage(content=question)
        history_messages = _dialog_dicts_to_lc_messages(dialog_messages)

        state: AgentState = {
            "messages": [system_message, *history_messages, user_message],
            "dialog_id": dialog_id,
            "input_is_safe": True,
        }

        _log_agent_context(
            dialog_id=dialog_id,
            system_message=system_message,
            history_messages=history_messages,
            user_message=user_message,
        )

        if log_agent_ctx:
            try:
                # Не логируем «сырые» dict'ы, только уже собранные сообщения для LangGraph.
                preview_messages = []
                for m in state["messages"]:
                    role = getattr(m, "type", getattr(m, "__class__", type(m)).__name__)
                    content = getattr(m, "content", "")
                    # Обрезаем контент, чтобы не засорять лог длинными документами.
                    if isinstance(content, str) and len(content) > 2000:
                        content_preview = content[:2000] + "... [truncated]"
                    else:
                        content_preview = content
                    preview_messages.append(
                        {
                            "role": role,
                            "content": content_preview,
                        }
                    )

                logger.info(
                    "Agent.run context (dialog_id=%s) state_dialog_id=%s messages=%s",
                    dialog_id,
                    state.get("dialog_id"),
                    preview_messages,
                )
            except Exception:
                logger.exception("Agent.run: failed to log agent context (dialog_id=%s)", dialog_id)

        token = is_general_document_question_cv.set(is_general_document_question)
        try:
            result = await self.graph.ainvoke(state)
        except Exception:
            logger.exception("Agent.run: graph.ainvoke failed (dialog_id=%s)", dialog_id)
            raise
        finally:
            is_general_document_question_cv.reset(token)

        messages = result["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage):
            return last_message.content

        return str(last_message.content) if hasattr(last_message, "content") else ""
