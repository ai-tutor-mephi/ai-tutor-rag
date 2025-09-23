# пайплайн поиска
# Запрос -> выделение аспектов -> qdrant поиск релевантных чанков -> выделение сущностей -> neo4j
# Для экономии средств для выделения сущностей и аспектов использовать в будущем бесплатную llm

import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase

import json
from typing import Optional



from Utils import FIND_CONTEXT, FIND_NODES, FIND_COMMUNITIES
from Prompts import ENTITY_SYS


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

light_model = os.getenv("MS_LIGHT_MODEL")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
ms = MsGraphRAG(driver=driver, model=light_model)


def dedup_keep_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = x.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out




async def extract_entities_names(chunks: list[str] | str, driver: GraphDatabase.driver=driver, ms: MsGraphRAG=ms) -> list[str]:
    """
    Extract entity names from the qdrant's chunks
    :param chunks:
    :param driver:
    :param ms:
    :return: entities
    """
    if isinstance(chunks, str):
        chunks = [chunks]


    try:
        ent_resp = await ms.achat(
            messages=[{"role": "system", "content": ENTITY_SYS},
                      {"role": "user", "content": " ".join(chunks)}],
            model=light_model
        )
        try:
            ents = json.loads(ent_resp.choices[0].message.content).get("entities", [])
            names = [e.get("name", "").strip() for e in ents if e.get("name")]
            return dedup_keep_order(names)
        except Exception:
            # попробуем из текста хоть что-то искать
            return dedup_keep_order(chunks)
    finally:
        ms.close()
        driver.close()


def _extract_data_from_graph(
    driver,
    names: list[str],
    doc_id: str,
    *,
    node_limit: int,
    edge_limit: int,
    k_hops: int = 1,
    database: str | None = None
) -> dict[str, any]:
    if not names:
        return {"centers": [], "edges": [], "communities": [], "context_text": ""}

    # 2.1 найдём подходящие узлы по любому читаемому полю
    with driver.session(database=database) as s:
        nodes_res = s.run(
            FIND_NODES,
            {"names": names, "node_limit": node_limit, "doc_id": doc_id}
        ).data()

        if not nodes_res:
            return {"centers": [], "edges": [], "communities": [], "context_text": ""}

        node_ids = [r["id"] for r in nodes_res]

        # 2.2 один хоп окружения (хочешь 2 — добавь ещё OPTIONAL MATCH)
        ctx = s.run(
            FIND_CONTEXT,
            {"ids": node_ids, "edge_limit": edge_limit, "doc_id": doc_id}
        ).data()

        # 2.3 (опционально) комьюнити
        comm = s.run(
            FIND_COMMUNITIES,
            {"ids": node_ids, "doc_id": doc_id}
        ).data()

    result = _assemble_context(ctx, comm, char_limit=None, include_nodes_without_summary=False)
    return result

def _assemble_context(
    ctx: list[dict[str, any]],
    comm: list[dict[str, any]],
    *,
    char_limit: Optional[int] = None,
    include_nodes_without_summary: bool = False
) -> dict[str, any]:
    """
    Из результатов графовых запросов (ctx, comm) собирает:
      - centers: уникальные центры с метками и summary
      - edges: список рёбер с соседями и summary
      - communities: список комьюнити
      - context_text: компактный текстовый контекст (опционально урезается по символам)

    ctx ожидается в формате строк запроса:
      center, center_labels, center_summary, rel, rel_summary, neighbor, neighbor_labels, neighbor_summary
    comm ожидается в формате:
      id, level, summary
    """
    lines: list[str] = []

    # --- центры ---
    centers: dict[str, dict[str, any]] = {}
    for row in ctx:
        c = row.get("center")
        if not c or c in centers:
            continue
        centers[c] = {
            "labels": row.get("center_labels") or [],
            "summary": row.get("center_summary"),
        }

    # В текст добавляем центр только если есть summary (или если явно разрешено)
    for name, data in centers.items():
        if data.get("summary") or include_nodes_without_summary:
            line = f"[NODE] {name}"
            if data.get("summary"):
                line += f" :: {data['summary']}"
            lines.append(line)

    # --- рёбра ---
    edges: list[dict[str, any]] = []
    for row in ctx:
        center, rel, neighbor = row.get("center"), row.get("rel"), row.get("neighbor")
        if not center or not rel or not neighbor:
            continue

        edges.append({
            "center": center,
            "rel": rel,
            "neighbor": neighbor,
            "rel_summary": row.get("rel_summary"),
            "neighbor_labels": row.get("neighbor_labels") or [],
            "neighbor_summary": row.get("neighbor_summary"),
        })

        edge_line = f"[EDGE] {center} -[{rel}]-> {neighbor}"
        if row.get("rel_summary"):
            edge_line += f" :: {row['rel_summary']}"
        if row.get("neighbor_summary"):
            edge_line += f" | neighbor_summary: {row['neighbor_summary']}"
        lines.append(edge_line)

    # --- комьюнити ---
    communities: list[dict[str, any]] = []
    for c in comm:
        item = {
            "id": c.get("id"),
            "level": c.get("level"),
            "summary": c.get("summary"),
        }
        communities.append(item)
        if item.get("summary"):
            lines.append(f"[COMMUNITY L{item['level']} #{item['id']}] {item['summary']}")

    # --- текст ---
    context_text = "\n".join(lines)
    if char_limit is not None and len(context_text) > char_limit:
        context_text = context_text[:char_limit]

    return {
        "centers": [{"name": k, **v} for k, v in centers.items()],
        "edges": edges,
        "communities": communities,
        "context_text": context_text,
    }

async def graph_context_from_chunks(
    chunks: list[str],
    doc_id: str,
    *,
    driver: GraphDatabase.driver=driver,
    k_hops: int = 1,
    node_limit: int = 50,
    edge_limit: int = 1000,
    context_lines_limit: int = 1000
) -> dict[str, any]:
    """
    Выделяет сущности из чанков, по ним ищет части графа и формирует контекст
    :param chunks:
    :param doc_id:
    :param driver:
    :param k_hops:
    :param node_limit:
    :param edge_limit:
    :param context_lines_limit:
    :return:
    """
    # 3.1 выделяем имена
    names = await extract_entities_names(chunks)
    if not names:
        return {"centers": [], "edges": [], "communities": [], "context_text": ""}

    # 3.2 подключаемся к нужной БД
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        result = _extract_data_from_graph(
            driver,
            names,
            doc_id,
            node_limit=node_limit,
            edge_limit=edge_limit,
            k_hops=k_hops,
            database=database
        )
        # подрежем текст если очень длинный
        if len(result.get("context_text","")) > context_lines_limit:
            result["context_text"] = result["context_text"][:context_lines_limit]
        return result
    finally:
        driver.close()


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(graph_context_from_chunks(["who is thomas?", "Thomas is working for what?", "Thomas"], doc_id="1")).get("context_text"))
