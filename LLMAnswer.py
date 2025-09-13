import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase


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

    sys_prompt = (
        "You are a graph-grounded assistant. "
        "Answer ONLY using the facts from the Graph Context. Use ALL relevant information. "
        "If information is missing, say you don't know."
        "In your answer, indicate only synthesized information. Do not indicate where you got it from and the connections between entities"
        "Ignore community summaries. Do not mention them in the answer."
    )
    user_prompt = f"Question: {question}\n\nGraph Context:\n{graph_context_text}"

    try:
        resp = await ms.achat(
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            model=os.getenv(ms_model_env_var)
        )
        answer = resp.content
        return answer
    finally:
        ms.close()
        driver.close()

if __name__ == "__main__":
    import asyncio
    print(asyncio.run(answer_with_graph("who is thomas?", "'context_text': '[NODE] THOMAS :: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[NODE] Thomas, Neo4j, and Grosuplje Community :: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[IN_COMMUNITY]-> Thomas, Neo4j, and Grosuplje Community | neighbor_summary: The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> NEO4J :: Thomas works for Neo4j | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[SUMMARIZED_RELATIONSHIP]-> GROSUPLJE :: Thomas lives in Grosuplje and attended school in Grosuplje. | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas went to school in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] THOMAS -[MENTIONS]-> Thomas lives in Grosuplje\n[EDGE] THOMAS -[RELATIONSHIP]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] THOMAS -[MENTIONS]-> Thomas works for Neo4j\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas lives in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas went to school in Grosuplje -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> NEO4J | neighbor_summary: Neo4j is a graph database company that provides a platform for building graph-based applications\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas, Neo4j, and Grosuplje Community -[IN_COMMUNITY]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> GROSUPLJE | neighbor_summary: Grosuplje is a town in Slovenia where Thomas attended school and currently resides.\n[EDGE] Thomas works for Neo4j -[MENTIONS]-> THOMAS | neighbor_summary: Thomas is a person who attended school in Grosuplje, works as an employee at Neo4j, and resides in Grosuplje.\n[COMMUNITY L0 #4:f8778c7d-506a-41b6-96fc-1ccc9c56b540:5] The community centers on Thomas, a resident of Grosuplje who works for Neo4j, a graph database company. Thomas’s ties to both the local town and his employer form the core of the community’s relational structure, with no additional entities or complex interactions identified.'")))
