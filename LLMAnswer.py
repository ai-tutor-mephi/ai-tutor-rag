import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase
from openai import OpenAI
from Utils import CONTEXT_SYS, REWRITE_QUESTION_SYS

load_dotenv()
# model = os.getenv("MS_GRAPHRAG_MODEL")

async def answer_with_graph(
    question: str,
    graph_context_text: str,
    *,
    ms_model_env_var: str = "MS_GRAPHRAG_MODEL"
) -> str:
    """
    берём исходный вопрос пользователя + собранный графовый контекст и отвечаем строго по нему.
    """
    # Инициализируем MsGraphRAG только для генерации ответа (никакой экстракции сущностей внутри!)
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    ms = MsGraphRAG(driver=driver, model=os.getenv(ms_model_env_var, "openai/gpt-oss-20b"))


    user_prompt = f"Question: {question}\n\nGraph Context:\n{graph_context_text}"

    try:
        resp = await ms.achat(
            messages=[{"role": "system", "content": CONTEXT_SYS},
                      {"role": "user", "content": user_prompt}],
            model=os.getenv(ms_model_env_var)
        )
        answer = resp.content
        return answer
    finally:
        ms.close()
        driver.close()

def rewrite_question_from_dialogue(question: str, dialogue:str) -> str:
    """
    Перефразирует вопрос на основе всего диалога
    :param question:
    :param dialogue:
    :return:
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))
    resp = client.chat.completions.create(
        model=os.getenv("MS_LIGHT_MODEL"),
        messages=[
            {"role": "system", "content": REWRITE_QUESTION_SYS},
            {"role": "user", "content": f"Dialogue: {dialogue}\nQuestion: {question}"}
        ]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    import asyncio
    print(asyncio.run(answer_with_graph("who is thomas?", "'context_text': '[NODE] THOMAS :: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[NODE] Thomas, Neo4j, and Grosuplje Community :: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[IN_COMMUNITY]-> Thomas, Neo4j, and Grosuplje Community | neighbor_summary: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> NEO4J :: Thomas works for Neo4j | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> GROSUPLJE :: Thomas lives in Grosuplje and attended school in Grosuplje. | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas went to school in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[MENTIONS]-> Thomas lives in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas works for Neo4j\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[COMMUNITY L0 #4:f8778c7d-506a-41b6-96fc-1ccc9c56b540:5] The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.'")))
