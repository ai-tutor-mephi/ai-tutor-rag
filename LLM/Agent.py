"""
Модуль для агента на основе LangGraph

"""

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Optional, Literal
import logging
import requests

import os
from dotenv import load_dotenv

from utils.Tools import tools
from utils.MyLogs import setup_logger
from LLM.LLMAnswer import LLM
from LLM.Prompts import AGENT_CONTEXT_SYS

load_dotenv()

# Настройка логов
logger = setup_logger(__file__)


class AgentState(TypedDict):
    """
    Состояние агента для LangGraph.
    """
    messages: Annotated[list, add_messages]
    dialog_id: str
    dialog_messages: List[Dict]
    input_is_safe: bool

def reject_input_node(state: AgentState) -> AgentState:
    """
    Узел для отклонения небезопасных входных сообщений.
    """
    messages = state["messages"]
    logger.debug("reject_input_node: rejecting unsafe input (dialog_id=%s)", state.get("dialog_id"))

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
    """
    Проверяет входное сообщение пользователя и сохраняет результат в state.
    """
    messages = state["messages"]
    last_message = messages[-1]

    user_text = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # Если guardrails server не настроен, то не делаем сетевых запросов (например, в unit-тестах).
    server_url = os.getenv("GUARDRAILS_SERVER_URL")

    if not server_url:
        logger.debug("input_guard_node: GUARDRAILS_SERVER_URL не задан => allow (dialog_id=%s)", state.get("dialog_id"))
        return {"input_is_safe": True}
    
    try:
        validate_url = f"{server_url.rstrip('/')}/guards/input_guard/validate"
        logger.debug("input_guard_node: POST %s", validate_url)
        response = requests.post(
            validate_url,
            json={"llmOutput": user_text},
            timeout=5,
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
            logger.exception("input_guard_node: failed to parse JSON (dialog_id=%s, url=%s)", state.get("dialog_id"), validate_url)
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
    """
    Определяет, куда переходить после проверки input_guard.
    """
    # По умолчанию считаем ввод безопасным, если флаг не установлен
    is_safe = state.get("input_is_safe", True)
    return "agent" if is_safe else "reject_input"

def create_agent_node(llm: ChatOpenAI, tools: list):
    """
    Создает узел агента, который вызывает LLM с инструментами.
    
    Args:
        llm: Языковая модель с привязанными инструментами
        tools: Список инструментов
        
    Returns:
        Функция узла агента
    """
    llm_with_tools = llm.bind_tools(tools)
    
    async def agent_node(state: AgentState):
        """
        Узел агента: вызывает LLM с инструментами.
        """
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Определяет, нужно ли вызывать инструменты или завершить работу.
    
    Args:
        state: Текущее состояние агента
        
    Returns:
        "tools" - если нужно вызвать инструменты
        "end" - если можно завершить работу
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Если последнее сообщение содержит tool_calls, вызываем инструменты
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Иначе завершаем работу
    return "end"


class Agent:
    """
    Агент для обработки запросов через LangGraph с использованием инструментов.
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """
        Инициализация агента.
        
        Args:
            llm: Экземпляр LLM (не используется).
        """
        # Инициализируем ChatOpenAI для работы с инструментами
        # Используем Groq API по умолчанию
        self.chat_llm = ChatOpenAI(
            model=os.getenv("MS_GRAPHRAG_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            temperature=0
        )
        
        # Создаем узел агента с инструментами
        self.agent_node = create_agent_node(self.chat_llm, tools)
        
        # Создаем узел для выполнения инструментов
        self.tool_node = ToolNode(tools)
        
        # Инициализируем граф
        self.graph = self.init_graph()
    
    def init_graph(self):
        """
        Инициализирует граф обработки запросов с условными переходами.
        
        Returns:
            Скомпилированный граф LangGraph
        """
        # Создаем граф с состоянием AgentState
        graph = StateGraph(AgentState)
        
        # Добавляем узлы
        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", self.tool_node)
        graph.add_node("input_guard", input_guard_node)
        graph.add_node("reject_input", reject_input_node)
        logger.info("Agent graph initialized (dialog_id will map to thread_id)")

        # Сначала проверяем ввод через guardrail
        graph.add_edge(START, "input_guard")

        # Ветвление после проверки: либо к агенту, либо к сообщению-отказу
        graph.add_conditional_edges(
            "input_guard",
            route_after_guard,
            {
                "agent": "agent",
                "reject_input": "reject_input",
            },
        )

        # После сообщения-отказа завершаем выполнение графа
        graph.add_edge("reject_input", END)
        
        # Условный переход: агент решает, вызывать ли инструменты
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # После выполнения инструментов возвращаемся к агенту
        graph.add_edge("tools", "agent")
        
        # Добавляем checkpoint для сохранения состояния
        memory = MemorySaver()
        
        return graph.compile(checkpointer=memory)

    async def run(
        self,
        question: str,
        dialog_id: str,
        dialog_messages: List[Dict]
    ) -> str:
        """
        Запускает обработку запроса через граф.
        
        Args:
            question: Вопрос пользователя
            dialog_id: Идентификатор диалога
            dialog_messages: История диалога
            
        Returns:
            Сгенерированный ответ
        """
        logger.debug(
            "Agent.run start (dialog_id=%s, question_len=%s)",
            dialog_id,
            len(question) if question is not None else None,
        )
        # Формируем системное сообщение на основе AGENT_CONTEXT_SYS
        # Добавляем dialog_id и историю диалога
        system_content = AGENT_CONTEXT_SYS + f"""

The rag_tool requires the dialog_id parameter. Use this dialog ID: {dialog_id}
"""
        
        # Добавляем историю диалога, если есть
        if dialog_messages:
            dialogue_history = "\n".join([
                f"{msg['role']}: {msg['message']}"
                for msg in dialog_messages
            ])
            system_content += f"\n\nDialogue history:\n{dialogue_history}"
        
        system_message = SystemMessage(content=system_content)
        
        # Создаем сообщение пользователя с вопросом
        user_message = HumanMessage(content=question)
        
        # Инициализируем начальное состояние
        state: AgentState = {
            "messages": [system_message, user_message],
            "dialog_id": dialog_id,
            "dialog_messages": dialog_messages
        }
        
        # Запускаем граф
        config = {"configurable": {"thread_id": dialog_id}}
        try:
            result = await self.graph.ainvoke(state, config)
        except Exception:
            logger.exception("Agent.run: graph.ainvoke failed (dialog_id=%s)", dialog_id)
            raise
        
        # Извлекаем финальный ответ из сообщений
        messages = result["messages"]
        last_message = messages[-1]
        
        # Если последнее сообщение - от AI, возвращаем его содержимое
        if isinstance(last_message, AIMessage):
            return last_message.content
        
        # Иначе возвращаем содержимое последнего сообщения
        return str(last_message.content) if hasattr(last_message, 'content') else ""
