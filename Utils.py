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

ENTITY_SYS = (
    "Extract named entities mentioned in the user question. "
    "Return strict JSON with fields: entities:[{name:string, type:string}] "
    "Use canonical short names presentable for lookup in a knowledge graph. "
    "No prose, JSON ONLY."
)

ASPECTS_SYS= """You are an assistant that extracts key aspects from a user query. 
Break the query into separate, minimal aspects that represent different topics, entities, or sub-questions. 
Output only the list of aspects, separated by the symbol "||". 
Do not add explanations, numbers, or extra text.

Example: 
Question: "Who is the CEO of Microsoft? What dog is eatting?"
Aspects: Who is the CEO of Microsoft || What dog is eating?

Question: "who is Jeff Bezos and what he doing, how much money he has?"
Aspects: Who is Jeff Bezos || What Jeff Bezos is doing || How much money Jeff Bezos has
"""

CONTEXT_SYS = (
    "You are a graph-grounded assistant. "
    "Answer ONLY using the facts from the Graph Context. Use ALL relevant information. "
    "If information is missing, say you don't know."
    "In your answer, indicate only synthesized information. Do not indicate where you got it from and the connections between entities"
    "Ignore community summaries. Do not mention them in the answer."
)

REWRITE_QUESTION_SYS="""
You are given a conversation history (dialogue) and the latest user question.  
Your task: rewrite the user question so that it becomes fully self-contained,  
using information from the dialogue. Keep the meaning, but remove ambiguity.  

Return only the rewritten question.(without "Rewritten Question". Only question)  

Examples:

Dialogue:  
User: "Who is Bush?"  
Assistant: "Bush was the 43rd President of the United States."  
Question: "What did he do?"  
Rewritten Question: "What did Bush do?"

---

Dialogue:  
User: "Where was Einstein born?"  
Assistant: "Einstein was born in Ulm, Germany."  
Question: "When did he die?"  
Rewritten Question: "When did Einstein die?"

---

Dialogue:  
User: "Tell me about Microsoft."  
Assistant: "Microsoft is a technology company founded by Bill Gates and Paul Allen."  
Question: "Who founded it?"  
Rewritten Question: "Who founded Microsoft?"
"""