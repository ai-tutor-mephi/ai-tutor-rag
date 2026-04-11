"""
Модуль для агента на основе LangGraph
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from LLM.LLMAnswer import LLM
from LLM.Nodes.Helpers import _dialog_dicts_to_lc_messages, _make_logged_tools_node
from LLM.Nodes.Nodes import (
    AgentState,
    create_agent_node,
    input_guard_node,
    reject_input_node,
    route_after_guard,
    should_continue,
)
from LLM.Prompts import AGENT_CONTEXT_SYS
from utils.MyLogs import setup_logger
from utils.Tools import tools

load_dotenv()

logger = setup_logger(__file__)


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
    ) -> str:
        logger.info(
            "Agent.run start dialog_id=%s question_len=%s history_turns=%s",
            dialog_id,
            len(question) if question is not None else None,
            len(dialog_messages) if dialog_messages else 0,
        )
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

        try:
            result = await self.graph.ainvoke(state)
        except Exception:
            logger.exception("Agent.run: graph.ainvoke failed (dialog_id=%s)", dialog_id)
            raise

        messages = result["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage):
            return last_message.content

        return str(last_message.content) if hasattr(last_message, "content") else ""
