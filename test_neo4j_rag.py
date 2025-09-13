import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase

import asyncio
import json

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

model = os.getenv("MS_GRAPHRAG_MODEL")

async def main():
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        uri,
        auth=(user, password)
    )

    # Initialize MsGraphRAG
    ms_graph = MsGraphRAG(driver=driver, model=model)

    # Define example texts and entity types
    example_texts = [
        "Tomaz works for Neo4j",
        "Tomaz lives in Grosuplje",
        "Tomaz went to school in Grosuplje"
    ]
    allowed_entities = ["Person", "Organization", "Location"]

    try:
        # Extract entities and relationships
        result = await ms_graph.extract_nodes_and_rels(example_texts, allowed_entities)
        print(result)

        # Generate summaries for nodes and relationships
        result = await ms_graph.summarize_nodes_and_rels()
        print(result)

        # Identify and summarize communities
        result = await ms_graph.summarize_communities()
        print(result)

    except Exception as e:
        print(e, "Data is in DB")

    # Close the connection
    ms_graph.close()

ENTITY_SYS = (
    "Extract named entities mentioned in the user question. "
    "Return strict JSON with fields: entities:[{name:string, type:string}] "
    "Use canonical short names presentable for lookup in a knowledge graph. "
    "No prose, JSON ONLY."
)

def dedup_keep_order(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

async def answer_from_graph(question: str, k_neighbors: int = 1):
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    ms = MsGraphRAG(driver=driver, model=os.getenv("MS_GRAPHRAG_MODEL", "openai/gpt-oss-20b"))

    # 1) Извлекаем сущности LLM-ом (JSON-формат)
    ent_resp = await ms.achat(
        messages=[{"role":"system","content":ENTITY_SYS},
                  {"role":"user","content":question}],
        model=os.getenv("MS_GRAPHRAG_MODEL")
    )
    try:
        ents = json.loads(ent_resp.choices[0].message.content).get("entities", [])
    except Exception:
        ents = []

    names = dedup_keep_order([e.get("name","").strip() for e in ents if e.get("name")])
    # Fallback, если LLM не дал ничего — попробуем искать по фразе вопроса
    if not names:
        names = [question]  # простой запасной вариант

    # 2) Тянем контекст из графа для всех найденных сущностей
    with driver.session() as s:
        # Найдём узлы по имени (без учёта регистра, подстрока)
        nodes = s.run("""
            UNWIND $names AS q
            MATCH (n)
            WHERE n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower(q)
            RETURN DISTINCT n LIMIT 20
        """, {"names": names}).data()

        node_ids = [rec["n"].element_id for rec in nodes]
        if not node_ids:
            # Можно добавить тут fulltext или vector fallback при желании
            return "Не нашёл сущностей в графе для этого вопроса."

        # Окружение 1 хоп (можно сделать k_neighbors=2 для 2-х хопов)
        ctx = s.run(f"""
            MATCH (n) WHERE elementId(n) IN $ids
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.name AS center, labels(n) AS center_labels, n.summary AS center_summary,
                   type(r) AS rel, r.summary AS rel_summary,
                   m.name AS neighbor, labels(m) AS neighbor_labels, m.summary AS neighbor_summary
            LIMIT 500
        """, {"ids": node_ids}).data()

        # Комьюнити (если есть)
        comm = s.run("""
            MATCH (n) WHERE elementId(n) IN $ids
            MATCH (n)-[:IN_COMMUNITY]->(c)
            RETURN DISTINCT c.communityId AS id, c.level AS level, c.summary AS summary
            ORDER BY level, id
        """, {"ids": node_ids}).data()

    # 3) Сборка компактного контекста
    lines = []
    # центрические summary
    by_center = {}
    for row in ctx:
        key = row["center"]
        if key and row.get("center_summary") and key not in by_center:
            by_center[key] = row["center_summary"]
    for name, summ in by_center.items():
        lines.append(f"[NODE] {name} :: {summ}")

    # связи
    for row in ctx:
        if not row["rel"] or not row["neighbor"]:
            continue
        line = f"[EDGE] {row['center']} -[{row['rel']}]-> {row['neighbor']}"
        if row.get("rel_summary"):
            line += f" :: {row['rel_summary']}"
        if row.get("neighbor_summary"):
            line += f" | neighbor_summary: {row['neighbor_summary']}"
        lines.append(line)

    # комьюнити
    for c in comm:
        if c.get("summary"):
            lines.append(f"[COMMUNITY L{c['level']} #{c['id']}] {c['summary']}")

    context = "\n".join(lines[:800])  # при желании ограничь объём

    # 4) Генерация ответа строго по контексту
    sys_prompt = (
        "You are a graph-grounded assistant. "
        "Answer ONLY using the facts from the Graph Context. "
        "If information is missing, say you don't know."
    )
    user_prompt = f"Question: {question}\n\nGraph Context:\n{context}"

    resp = await ms.achat(
        messages=[{"role":"system","content":sys_prompt},
                  {"role":"user","content":user_prompt}],
        model=os.getenv("MS_GRAPHRAG_MODEL")
    )
    answer = resp.choices[0].message.content


    ms.close()

    driver.close()
    return answer



if __name__ == "__main__":
    asyncio.run(main())
    print(asyncio.run(answer_from_graph("who is thomas?")))


# Прикрутить векторную базу данных для поиска сущностей по запросу наиболее подходящих
