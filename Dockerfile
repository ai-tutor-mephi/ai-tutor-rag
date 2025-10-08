FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Создаем директорию для моделей ---
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p $HF_HOME

# --- Предзагружаем модель, чтобы закэшировалась в образе ---
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    model_id='BAAI/bge-m3'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModel.from_pretrained(model_id)"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag:rag", "--host", "0.0.0.0", "--port", "8000"]

