FIND_NODES = """
UNWIND $names AS q
MATCH (n)
WHERE n.doc_id = $doc_id
WITH q, n, [
  n.name, n.title, n.text, n.summary
] AS fields
WITH q, n, [v IN fields WHERE v IS NOT NULL] AS vals
WHERE any(val IN vals WHERE toLower(toString(val)) CONTAINS toLower(q))
RETURN DISTINCT elementId(n) AS id
LIMIT $node_limit
"""


FIND_CONTEXT = """
MATCH (n)
WHERE elementId(n) IN $ids AND n.doc_id = $doc_id
OPTIONAL MATCH (n)-[r {doc_id: $doc_id}]-(m {doc_id: $doc_id})
RETURN coalesce(n.name, n.title, n.text, "") AS center,
       labels(n) AS center_labels,
       n.summary AS center_summary,
       type(r) AS rel, r.summary AS rel_summary,
       coalesce(m.name, m.title, m.text, "") AS neighbor,
       labels(m) AS neighbor_labels,
       m.summary AS neighbor_summary
LIMIT $edge_limit
"""


FIND_COMMUNITIES = """
MATCH (n)
WHERE elementId(n) IN $ids AND n.doc_id = $doc_id
OPTIONAL MATCH (n)-[:IN_COMMUNITY]->(c)
WHERE c.doc_id = $doc_id
WITH DISTINCT c WHERE c IS NOT NULL
RETURN elementId(c) AS id,
       c.level      AS level,
       coalesce(c.summary, "") AS summary
ORDER BY level, id
"""

