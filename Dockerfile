FROM python:3.12-slim

WORKDIR /app

# --- Устанавливаем CPU-версию PyTorch ---
RUN pip install --no-cache-dir torch==2.8.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Создаем директорию для моделей ---
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p $HF_HOME

# --- Предзагружаем модель ---
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    from transformers import AutoTokenizer, AutoModel; \
    model_id='BAAI/bge-m3'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModel.from_pretrained(model_id)"

COPY ms_graphrag_neo4j ./ms_graphrag_neo4j
RUN pip install -e ./ms_graphrag_neo4j

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag:rag", "--host", "0.0.0.0", "--port", "8000"]

