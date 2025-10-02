import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase
from openai import OpenAI
from Prompts import CONTEXT_SYS, REWRITE_QUESTION_SYS

import logging
from pathlib import Path

# путь к директории с текущим файлом
base_dir = Path(__file__).resolve().parent

# подняться на n директорий вверх
root_dir = base_dir.parents[1]

# путь к Logs
logs_dir = root_dir / "Logs"
logs_dir.mkdir(parents=True, exist_ok=True)  # создаём папку, если её нет

# сам лог-файл
log_file = logs_dir / "LLM.log"

logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    filemode="a", 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()
model = os.getenv("MS_GRAPHRAG_MODEL")
light_model=os.getenv("MS_LIGHT_MODEL")

# Инициализируем MsGraphRAG только для генерации ответа (никакой экстракции сущностей внутри!) Можно заменить позже на OpenAi
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
ms = MsGraphRAG(driver=driver, model=os.getenv("MS_GRAPHRAG_MODEL", "openai/gpt-oss-20b"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))

class LLM:
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
        берём исходный вопрос пользователя + собранный графовый контекст и отвечаем строго по нему.
        """

        user_prompt = f"Question: {question}\n\nGraph Context:\n{graph_context_text}"

        logging.info(f"Обращаемся с промптом: {user_prompt}\n к модели {self.model}")
        resp = await self.ms.achat(
            messages=[{"role": "system", "content": CONTEXT_SYS},
                    {"role": "user", "content": user_prompt}],
            model=self.model
        )
        answer = resp.content
        logging.info(f"Ответ модели: {answer}")
        return answer



    async def rewrite_question_from_dialogue(self, question: str, dialogue:str) -> str:
        """
        Перефразирует вопрос на основе всего диалога
        :param question:
        :param dialogue:
        :return:
        """

        logging.info(f"Перефразируем вопрос: {question}\n на основе диалога: {dialogue}\n модель: {self.light_model}")
        resp = self.client.chat.completions.create(
            model=self.light_model,
            messages=[
                {"role": "system", "content": REWRITE_QUESTION_SYS},
                {"role": "user", "content": f"Dialogue: {dialogue}\nQuestion: {question}"}
            ]
        )
        answer = resp.choices[0].message.content

        logging.info(f"Перефразированный вопрос: {answer}")
        return answer


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(LLM.answer_with_graph("who is thomas?", "'context_text': '[NODE] THOMAS :: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[NODE] Thomas, Neo4j, and Grosuplje Community :: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[IN_COMMUNITY]-> Thomas, Neo4j, and Grosuplje Community | neighbor_summary: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> NEO4J :: Thomas works for Neo4j | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> GROSUPLJE :: Thomas lives in Grosuplje and attended school in Grosuplje. | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas went to school in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[MENTIONS]-> Thomas lives in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas works for Neo4j\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[COMMUNITY L0 #4:f8778c7d-506a-41b6-96fc-1ccc9c56b540:5] The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.'")))
