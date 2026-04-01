FROM python:3.12-slim

WORKDIR /app

# build arg для токена Guardrails Hub
ARG GUARDRAILS_TOKEN

# базовые зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# обновляем pip
RUN pip install --no-cache-dir --upgrade pip

# PyTorch CPU
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# зависимости проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p $HF_HOME

# предзагрузка модели
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
from transformers import AutoTokenizer, AutoModel; \
model_id='BAAI/bge-m3'; \
AutoTokenizer.from_pretrained(model_id); \
AutoModel.from_pretrained(model_id)"

# локальный модуль
COPY ms_graphrag_neo4j ./ms_graphrag_neo4j
RUN pip install -e ./ms_graphrag_neo4j

# остальной код
COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag:rag", "--host", "0.0.0.0", "--port", "8000"]