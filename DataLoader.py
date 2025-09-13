# для каждого чанка, загруженного в qdrant и neo4j надо одинаковый чтобы в обоих БД у него был одинаковый id

import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase


import json

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

model = os.getenv("MS_GRAPHRAG_MODEL")

def extract_data_from_json(data: json) -> list:
    pass

async def create_graph(data: json, uri: str = neo4j_uri,
                       user: str = neo4j_user, password: str = neo4j_password, model: str = model) -> None:
    """
    Connect to Neo4j and create a graph from the given data.
    :param data: data to be added to the graph
    :param uri: graph database uri
    :param user: graph database user
    :param password: graph database password
    :return:
    """

    # Connect to Neo4j
    driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    # Initialize MsGraphRAG
    ms_graph = MsGraphRAG(driver=driver, model=model)

    # data: list = extract_data_from_json(data)

    try:
        # Extract entities and relationships
        result = await ms_graph.extract_nodes_and_rels(data, []) # вот тут будто ничего не генерирует. я уже изменил промпт в prompth.py но я хз
        print(result)

        # Generate summaries for nodes and relationships
        result = await ms_graph.summarize_nodes_and_rels()
        print(result)

        # Identify and summarize communities
        result = await ms_graph.summarize_communities()
        print(result)


    except Exception:
        import traceback
        traceback.print_exc()

    # Close the connection
    ms_graph.close()
    driver.close()


ENTITY_SYS = (
    "Extract named entities mentioned in the user question. "
    "Return strict JSON with fields: entities:[{name:string, type:string}] "
    "Use canonical short names presentable for lookup in a knowledge graph. "
    "No prose, JSON ONLY."
)

def load_in_qdrant(data: json) -> None:
    pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_graph(["Thomas works for Neo4j", "Thomas lives in Grosuplje", "Thomas went to school in Grosuplje"]))
