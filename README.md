# RAG
[this](https://github.com/neo4j-contrib/ms-graphrag-neo4j/blob/main/.gitignore) repository used for this project. It was modified

Microservice for generating a response based on a question and a document.

### Ingestion
<img width="2048" height="1230" alt="image" src="https://github.com/user-attachments/assets/2bd6feac-4a1e-4c07-949e-c9a1d301f305" />

### Retrieval & Generation
1) extract aspects from the question.
2) For each aspect, search for the top_k closest chunks in the vector database.
3) Extract entities from them and search the graph database.
4) Based on the found graph, form a context and send it to LLM with a question

