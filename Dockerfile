FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# базовые зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY ms_graphrag_neo4j ./ms_graphrag_neo4j

RUN uv sync --frozen --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p $HF_HOME

# предзагрузка модели
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
from transformers import AutoTokenizer, AutoModel; \
model_id='BAAI/bge-m3'; \
AutoTokenizer.from_pretrained(model_id); \
AutoModel.from_pretrained(model_id)"

# остальной код
COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag:rag", "--host", "0.0.0.0", "--port", "8000"]
