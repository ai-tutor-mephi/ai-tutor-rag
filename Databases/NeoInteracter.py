
import os
from dotenv import load_dotenv

from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase

import json
from typing import Optional

from .FindsForNeo import FIND_CONTEXT, FIND_NODES, FIND_COMMUNITIES
from LLM.Prompts import ENTITY_SYS

import asyncio
import sys

import logging
from pathlib import Path

logs_dir = Path("/Logs")
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "neo.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

model = os.getenv("MS_GRAPHRAG_MODEL")
light_model = os.getenv("MS_LIGHT_MODEL")

database = os.getenv("NEO4J_DATABASE", "neo4j")

# Connect to Neo4j
driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_user, neo4j_password)
)

# Initialize MsGraphRAG
ms_graph = MsGraphRAG(driver=driver, model=model)


class NeoInteracter:
    def __init__(self, driver=driver, ms_graph=ms_graph, model=model, light_model=light_model, database=database):
        self.driver = driver
        self.ms_graph = ms_graph
        self.model = model
        self.light_model = light_model
        self.database = database

    def __del__(self):
        self.ms_graph.close()
        self.driver.close()

    async def create_graph(self, chunks: list[dict]) -> None:
    
        """
        Connect to Neo4j and create a graph from the given data.
        :param chunks: chunks to be added to the graph
        :return:
        """

        data = [d.get("text") for d in chunks]

        try:
            # Extract entities and relationships
            logging.info("Извлечение сущностей и связей...")
            result = await self.ms_graph.extract_nodes_and_rels(data, [])
            logging.info(f"Сущности и связи извлечены {result}")
            print(result)

            dialog_id = chunks[0].get("dialog_id")

            with self.driver.session(database=self.database) as session:
                session.run(
                    """
                    MATCH (n)
                    WHERE n.dialog_id IS NULL
                    SET n.dialog_id = $dialog_id
                    """,
                    dialog_id=dialog_id
                )
                session.run(
                    """
                    MATCH ()-[r]-()
                    WHERE r.dialog_id IS NULL
                    SET r.dialog_id = $dialog_id
                    """,
                    dialog_id=dialog_id
                )
                
            # Generate summaries for nodes and relationships
            logging.info("Генерация резюме для сущностей и связей...")
            result = await self.ms_graph.summarize_nodes_and_rels()
            print(result)

            # Identify and summarize communities
            logging.info("Выделение и суммаризация комьюнити...")
            result = await self.ms_graph.summarize_communities()
            print(result)


        except Exception as e:
            import traceback
            logging.error(f"Ошибка при создании графа в neo4j\n{e}")
            traceback.print_exc()

    @staticmethod
    def dedup_keep_order(items: list[str]) -> list[str]:
        seen, out = set(), []
        for x in items:
            x = x.strip()
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out

    @staticmethod
    async def extract_entities_names(chunks: list[str] | str, ms, light_model) -> list[str]:
        """
        Extract entity names from the qdrant's chunks
        :param chunks:
        :return: entities
        """
        if isinstance(chunks, str):
            chunks = [chunks]

        logging.info("Вызов модели для извлечения сущностей...")
        ent_resp = await ms.achat(
            messages=[{"role": "system", "content": ENTITY_SYS},
                    {"role": "user", "content": " ".join(chunks)}],
            model=light_model
        )
        ent_resp = ent_resp.content
        logging.info(f"Ответ модели для извлечения сущностей: {ent_resp}")

        try:
            parsed = json.loads(ent_resp)
            if isinstance(parsed, list):
                ents = parsed
            else:
                ents = parsed.get("entities", [])
            
            names = [e.get("name", "").strip() for e in ents if e.get("name")]
            return await asyncio.to_thread(NeoInteracter.dedup_keep_order, names)
        except Exception:
            # попробуем из текста хоть что-то искать
            logging.error("Ошибка при разборе ответа модели для извлечения сущностей")
            return await asyncio.to_thread(NeoInteracter.dedup_keep_order, chunks)

            
    @staticmethod
    async def _extract_data_from_graph(
        driver,
        names: list[str],
        dialog_id: str,
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
                {"names": names, "node_limit": node_limit, "dialog_id": dialog_id}
            ).data()

            if not nodes_res:
                return {"centers": [], "edges": [], "communities": [], "context_text": ""}

            node_ids = [r["id"] for r in nodes_res]

            # 2.2 один хоп окружения (хочешь 2 — добавь ещё OPTIONAL MATCH)
            ctx = s.run(
                FIND_CONTEXT,
                {"ids": node_ids, "edge_limit": edge_limit, "dialog_id": dialog_id}
            ).data()

            # 2.3 (опционально) комьюнити
            comm = s.run(
                FIND_COMMUNITIES,
                {"ids": node_ids, "dialog_id": dialog_id}
            ).data()

        result = await asyncio.to_thread(NeoInteracter._assemble_context, 
                                   ctx, comm, char_limit=None, 
                                   include_nodes_without_summary=False)
        return result

    @staticmethod
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
        self,
        chunks: list[str],
        dialog_id: str,
        *,
        k_hops: int = 1,
        node_limit: int = 50,
        edge_limit: int = 1000,
        context_lines_limit: int = 1000
    ) -> dict[str, any]:
        """
        Выделяет сущности из чанков, по ним ищет части графа и формирует контекст
        :param chunks:
        :param dialog_id:
        :param k_hops:
        :param node_limit:
        :param edge_limit:
        :param context_lines_limit:
        :return:
        """
        # 3.1 выделяем имена
        names = await self.extract_entities_names(chunks, ms=self.ms_graph, light_model=self.light_model)
        if not names:
            return {"centers": [], "edges": [], "communities": [], "context_text": ""}
        
        logging.info(f"Выделены имена для поиска в графе: {names}")

        # 3.2 подключаемся к нужной БД

        try:
            result = await NeoInteracter._extract_data_from_graph(
                self.driver,
                names,
                dialog_id,
                node_limit=node_limit,
                edge_limit=edge_limit,
                k_hops=k_hops,
                database=self.database
            )
            logging.info(f"Данные из графа получены: {result.get('context_text','')[:500]}...")

            # подрежем текст если очень длинный
            if len(result.get("context_text","")) > context_lines_limit:
                result["context_text"] = result["context_text"][:context_lines_limit]
            
            logging.info(f"Сформирован контекст из графа: {result.get('context_text','')[:500]}...")
            return result
        except Exception as e:
            return None # подумать над выводом из классов
        



async def main():
    data = [{"text": "Thomas works for Neo4j", "dialog_id": "1"},
                              {"text": "Thomas lives in Grosuplje", "dialog_id": "1"},
                              {"text": "Thomas went to school in Grosuplje", "dialog_id": "1"}]
    inter = NeoInteracter()
    await inter.create_graph(data)

    ans = await inter.graph_context_from_chunks(['who is thomas?', "Thomas is working for what?"], "1")

    print(ans)

if __name__=="__main__":
    asyncio.run(main())



