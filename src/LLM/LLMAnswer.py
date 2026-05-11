"""
Модуль для работы с языковыми моделями (LLM).

Этот модуль предоставляет интерфейс для:
- Генерации ответов на основе графового контекста
- Перефразирования вопросов с учетом истории диалога

Использует OpenAI API и MsGraphRAG для работы с графовыми данными.
"""

import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase
import json
import re
from openai import OpenAI
from .Prompts import CONTEXT_SYS, GENERAL_QUESTION_CLASSIFY_SYS, REWRITE_QUESTION_SYS

import logging

logger = logging.getLogger(__name__)

load_dotenv()
model = os.getenv("MS_GRAPHRAG_MODEL")
light_model = os.getenv("MS_LIGHT_MODEL")

# Инициализация подключений
# MsGraphRAG используется только для генерации ответов (не для экстракции сущностей)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
ms = MsGraphRAG(driver=driver, model=os.getenv("MS_GRAPHRAG_MODEL", "openai/gpt-oss-20b"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"))


class LLM:
    """
    Класс для работы с языковыми моделями.
    
    Предоставляет методы для:
    - Генерации ответов на основе графового контекста
    - Перефразирования вопросов с учетом истории диалога
    - Классификации «общий вопрос по всему документу» vs узкий запрос
    """
    def __init__(self, model=model, driver=driver, ms=ms, client=client, light_model=light_model):
        self.model=model
        self.light_model=light_model
        self.driver=driver
        self.ms=ms
        self.client=client

    def __del__(self):
        self.ms.close()
        self.driver.close()
    

    async def answer_with_graph(
        self,
        question: str,
        graph_context_text: str,
    ) -> str:
        """
        Генерирует ответ на вопрос на основе графового контекста.
        
        Использует LLM для генерации ответа, который строго основан
        на предоставленном графовом контексте из Neo4j.
        
        Args:
            question: Вопрос пользователя
            graph_context_text: Графовый контекст из Neo4j
            
        Returns:
            Сгенерированный ответ
        """

        user_prompt = f"Question: {question}\n\nGraph Context:\n{graph_context_text}"

        logger.info(f"Обращаемся с промптом: {user_prompt}\n к модели {self.model}")
        resp = await self.ms.achat(
            messages=[{"role": "system", "content": CONTEXT_SYS},
                    {"role": "user", "content": user_prompt}],
            model=self.model
        )
        answer = resp.content
        logger.info(f"Ответ модели: {answer}")
        return answer



    async def rewrite_question_from_dialogue(self, question: str, dialogue: str) -> str:
        """
        Перефразирует вопрос на основе истории диалога.
        
        Улучшает понимание вопроса, учитывая контекст предыдущих
        сообщений в диалоге. Это позволяет лучше интерпретировать
        вопросы, которые ссылаются на предыдущий контекст.
        
        Args:
            question: Исходный вопрос пользователя
            dialogue: История диалога в виде строки
            
        Returns:
            Перефразированный вопрос
        """

        logger.info(f"Перефразируем вопрос: {question}\n на основе диалога: {dialogue}\n модель: {self.light_model}")
        resp = self.client.chat.completions.create(
            model=self.light_model,
            messages=[
                {"role": "system", "content": REWRITE_QUESTION_SYS},
                {"role": "user", "content": f"Dialogue: {dialogue}\nQuestion: {question}"}
            ]
        )
        answer = resp.choices[0].message.content

        logger.info(f"Перефразированный вопрос: {answer}")
        return answer

    async def classify_general_document_question(self, question: str, dialogue: str) -> bool:
        """
        Определяет, требует ли вопрос обзора всего загруженного материала (конспект, все теоремы, о чём документ…).

        Returns:
            True если вопрос «общий» по документу, False если достаточно точечного поиска.
        """
        logger.info(
            "Классификация общий/узкий вопрос (dialogue_len=%s, question_preview=%s)",
            len(dialogue or ""),
            (question or "")[:200],
        )
        resp = self.client.chat.completions.create(
            model=self.light_model,
            messages=[
                {"role": "system", "content": GENERAL_QUESTION_CLASSIFY_SYS},
                {
                    "role": "user",
                    "content": f"Dialogue:\n{dialogue}\n\nQuestion:\n{question}",
                },
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        logger.info(f"Ответ классификатора: {raw}")

        try:
            m = re.search(r"\{[^}]*\"general\"\s*:\s*(true|false)[^}]*\}", raw, re.I)
            blob = m.group(0) if m else raw
            data = json.loads(blob)
            return bool(data.get("general"))
        except (json.JSONDecodeError, TypeError, AttributeError):
            logger.warning("Не удалось распарсить JSON классификатора, считаем вопрос узким")
            return False


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(LLM.answer_with_graph("who is thomas?", "'context_text': '[NODE] THOMAS :: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[NODE] Thomas, Neo4j, and Grosuplje Community :: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[IN_COMMUNITY]-> Thomas, Neo4j, and Grosuplje Community | neighbor_summary: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> NEO4J :: Thomas works for Neo4j | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> GROSUPLJE :: Thomas lives in Grosuplje and attended school in Grosuplje. | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas went to school in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[MENTIONS]-> Thomas lives in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas works for Neo4j\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[COMMUNITY L0 #4:f8778c7d-506a-41b6-96fc-1ccc9c56b540:5] The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.'")))
