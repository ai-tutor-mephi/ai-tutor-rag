# для каждого чанка, загруженного в qdrant и neo4j надо одинаковый чтобы в обоих БД у него был одинаковый id

import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

model = os.getenv("MS_GRAPHRAG_MODEL")

# Connect to Neo4j
driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_user, neo4j_password)
)

# Initialize MsGraphRAG
ms_graph = MsGraphRAG(driver=driver, model=model)

async def create_graph(chunks: list[dict], ms_graph: MsGraphRAG=ms_graph, driver: GraphDatabase.driver=driver) -> None:
    """
    Connect to Neo4j and create a graph from the given data.
    :param chunks: chunks to be added to the graph
    :param ms_graph: MsGraphRAG instance
    :param driver: Neo4j driver
    :return:
    """

    data = [d.get("text") for d in chunks]

    try:
        # Extract entities and relationships
        result = await ms_graph.extract_nodes_and_rels(data, [])
        print(result)

        # Generate summaries for nodes and relationships
        result = await ms_graph.summarize_nodes_and_rels()
        print(result)

        # Identify and summarize communities
        result = await ms_graph.summarize_communities()
        print(result)

        doc_id = chunks[0].get("doc_id")

        # Проставить doc_id всем новым нодам и рёбрам
        with driver.session() as session:
            session.run(
                """
                MATCH (n)
                WHERE NOT EXISTS(n.doc_id)
                SET n.doc_id = $doc_id
                """,
                doc_id=doc_id
            )
            session.run(
                """
                MATCH ()-[r]-()
                WHERE NOT EXISTS(r.doc_id)
                SET r.doc_id = $doc_id
                """,
                doc_id=doc_id
            )


    except Exception:
        import traceback
        traceback.print_exc()

    # Close the connection
    ms_graph.close()
    driver.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_graph([{"text": "Thomas works for Neo4j", "doc_id": "1"},
                              {"text": "Thomas lives in Grosuplje", "doc_id": "1"},
                              {"text": "Thomas went to school in Grosuplje", "doc_id": "1"}]))
