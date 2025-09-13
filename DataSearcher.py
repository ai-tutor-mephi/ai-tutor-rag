# пайплайн поиска
# Запрос -> выделение аспектов -> qdrant поиск релевантных чанков -> выделение сущностей -> neo4j
# Для экономии средств для выделения сущностей и аспектов использовать в будущем бесплатную llm

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

# def dedup_keep_order(items: list[str]) -> list[str]:
#     seen = set()
#     out = []
#     for x in items:
#         if x not in seen:
#             seen.add(x)
#             out.append(x)
#     return out

def extract_aspects_from_question(question: str) -> list[str]:
    """
    Extract aspects from the question
    :param question:
    :return:
    """
    pass


ENTITY_SYS = (
    "Extract named entities mentioned in the user question. "
    "Return strict JSON with fields: entities:[{name:string, type:string}] "
    "Use canonical short names presentable for lookup in a knowledge graph. "
    "No prose, JSON ONLY."
)

async def extract_entities_names(chunks: list[str] | str) -> list[str]:

    """
    Поиск дорогой, поэтому от векторной бд нужно прокидывать не более 5 чанков

    Функция подойдет для выделения сущностей из чанков

    Extract candidate names from chunks of text
    :param chunks: chunks of text
    :return: list of candidate names
    """
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )

    ms = MsGraphRAG(driver=driver, model=os.getenv("MS_GRAPHRAG_MODEL", "openai/gpt-oss-20b"))

    # для корректной отработки .join
    if type(chunks) == str:
        chunks = [chunks]

    ent_resp = await ms.achat(
        messages=[{"role": "system", "content": ENTITY_SYS},
                  {"role": "user", "content": " ".join(chunks)}],
        model=os.getenv("MS_GRAPHRAG_MODEL")
    )
    try:
        ents = json.loads(ent_resp.choices[0].message.content).get("entities", [])
    except Exception:
        return chunks  # если ничего llm не вернула

    finally:
        ms.close()
        driver.close()

    # return dedup_keep_order
    return [e.get("name", "").strip() for e in ents if e.get("name")]


async def graph_context_from_chunks(
    chunks: list[str],
    *,
    k_hops: int = 1,
    node_limit: int = 50,
    edge_limit: int = 1000,
    context_lines_limit: int = 800
) -> dict[str, any]:
    """
    Extract context from chunks of text
    :param chunks: chunks of text
    :param k_hops: number of hops
    :param node_limit:
    :param edge_limit:
    :param context_lines_limit:
    :return: context
    """
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    try:
        names = await extract_entities_names(chunks)
        if not names:
            return {
                "centers": [],
                "edges": [],
                "communities": [],
                "context_text": ""
            }

        with driver.session() as s:
            # 1) Находим кандидаты-узлы по подстроке (без учета регистра).
            nodes = s.run(
                """
                UNWIND $names AS q
                MATCH (n)
                WITH DISTINCT n, toLower(q) AS q
                WHERE toLower(
                        toString(
                          coalesce(n.name, n.title, n.text, n.summary, "")
                        )
                      ) CONTAINS q
                RETURN DISTINCT elementId(n) AS id
                LIMIT $node_limit
                """,
                {"names": names, "node_limit": node_limit}
            ).data()

            if not nodes:
                return {
                    "centers": [],
                    "edges": [],
                    "communities": [],
                    "context_text": ""
                }

            node_ids = [rec["id"] for rec in nodes]

            # 2) Окружение K-хопов (сейчас 1). Для 2-х хопов — добавить ещё OPTIONAL MATCH слоя.
            #   Возвращаем узлы-центры, рёбра и соседей, плюс summary, если есть.
            ctx = s.run(
                """
                MATCH (n)
                WHERE elementId(n) IN $ids
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN coalesce(n.name, n.title, n.text, "") AS center,
                       labels(n) AS center_labels,
                       n.summary AS center_summary,
                       type(r) AS rel, r.summary AS rel_summary,
                       coalesce(m.name, m.title, m.text, "") AS neighbor,
                       labels(m) AS neighbor_labels,
                       m.summary AS neighbor_summary
                LIMIT $edge_limit
                """,
                {"ids": node_ids, "edge_limit": edge_limit}
            ).data()

            # 3) Комьюнити (если размечены)
            comm = s.run(
                """
                MATCH (n) 
                WHERE elementId(n) IN $ids
                OPTIONAL MATCH (n)-[:IN_COMMUNITY]->(c)
                WITH DISTINCT c
                WHERE c IS NOT NULL
                RETURN 
                  elementId(c) AS id,
                  c.level      AS level,
                  coalesce(c.summary, "") AS summary
                ORDER BY level, id
                """,
                {"ids": node_ids}
            ).data()

        # 4) Сборка компактного текстового контекста
        lines: list[str] = []
        centers = {}
        for row in ctx:
            c = row.get("center")
            if c and row.get("center_summary") and c not in centers:
                centers[c] = {
                    "labels": row.get("center_labels") or [],
                    "summary": row.get("center_summary")
                }
        for name, data in centers.items():
            lines.append(f"[NODE] {name} :: {data['summary']}")

        edges = []
        for row in ctx:
            rel = row.get("rel")
            neighbor = row.get("neighbor")
            center = row.get("center")
            if not center or not rel or not neighbor:
                continue
            edge_line = f"[EDGE] {center} -[{rel}]-> {neighbor}"
            if row.get("rel_summary"):
                edge_line += f" :: {row['rel_summary']}"
            if row.get("neighbor_summary"):
                edge_line += f" | neighbor_summary: {row['neighbor_summary']}"
            edges.append({
                "center": center,
                "rel": rel,
                "neighbor": neighbor,
                "rel_summary": row.get("rel_summary"),
                "neighbor_labels": row.get("neighbor_labels") or [],
                "neighbor_summary": row.get("neighbor_summary")
            })
            lines.append(edge_line)

        communities = []
        for c in comm:
            if c.get("summary"):
                lines.append(f"[COMMUNITY L{c['level']} #{c['id']}] {c['summary']}")
            communities.append({
                "id": c.get("id"),
                "level": c.get("level"),
                "summary": c.get("summary")
            })

        context_text = "\n".join(lines[:context_lines_limit])

        return {
            # "centers": [{"name": k, **v} for k, v in centers.items()],
            # "edges": edges,
            # "communities": communities,
            "context_text": context_text
        }
    finally:
        driver.close()


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(graph_context_from_chunks(["who is thomas?", "Thomas is working for what?", "Thomas"])))
