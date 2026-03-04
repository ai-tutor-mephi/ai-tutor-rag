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


import os
from dotenv import load_dotenv

from utils.Tools import tools
from utils.MyLogs import setup_logger
from LLM.LLMAnswer import LLM
from LLM.Prompts import AGENT_CONTEXT_SYS

load_dotenv()

# Настройка логов
setup_logger(__file__)


class AgentState(TypedDict):
    """
    Состояние агента для LangGraph.
    """
    messages: Annotated[list, add_messages]
    dialog_id: str
    dialog_messages: List[Dict]


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
        
        # Определяем поток выполнения
        graph.add_edge(START, "agent")
        
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
        result = await self.graph.ainvoke(state, config)
        
        # Извлекаем финальный ответ из сообщений
        messages = result["messages"]
        last_message = messages[-1]
        
        # Если последнее сообщение - от AI, возвращаем его содержимое
        if isinstance(last_message, AIMessage):
            return last_message.content
        
        # Иначе возвращаем содержимое последнего сообщения
        return str(last_message.content) if hasattr(last_message, 'content') else ""
