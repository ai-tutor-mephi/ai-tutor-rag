# RAG
[this](https://github.com/neo4j-contrib/ms-graphrag-neo4j/blob/main/.gitignore) repository used for this project. It was modified

Microservice for generating a response based on a question and a document.

## Презентация сервиса AI Tutor
Для полного просмотра кликните по картинке

<div align="center">
  <a href="./docs/AI_Tutor_presentation.pdf">
    <img src="./docs/ai-tutor-preview.png" alt="AI Tutor Presentation" width="800">
  </a>
</div>

### Ingestion
<img width="2048" height="1230" alt="image" src="https://github.com/user-attachments/assets/2bd6feac-4a1e-4c07-949e-c9a1d301f305" />

### Retrieval & Generation
1) extract aspects from the question.
2) For each aspect, search for the top_k closest chunks in the vector database.
3) Extract entities from them and search the graph database.
4) Based on the found graph, form a context and send it to LLM with a question


# Quick start:
1) Clone the repository and change your directory to it
2) Write your own ```.env``` with that fields:
```
GROQ_API_KEY=

QDRANT_KEY=
QDRANT_URL=

NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=

OPENAI_API_KEY=
OPENAI_BASE_URL=
MS_GRAPHRAG_MODEL=openai/gpt-oss-20b
MS_LIGHT_MODEL=llama-3.1-8b-instant
```
3) Write ```docker compose up --build``` that build the docker-image and up it in your device

# Endpoints

### ```rag/load``` 
Allows you to load data into the service. 
JSON's body:
```
{
  "content": [
              {
              "fileId": "...",
              "fileName": "...", 
              "text": "..."
              }
  ],
  "dialogId": "..."
}
```
### ```rag/query```
Allows you to give an answer to your question
JSON's body:
```
{
  "dialogId": "...",
  "dialogMessages": [
          {
          "message": "...", 
          "role": "..."
          },
          ...
  ],
  "question": "..."
  }
```

