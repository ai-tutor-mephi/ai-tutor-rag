FROM python:3.12-slim

WORKDIR /app

# 1. Сначала ставим системные зависимости (если нужны) и обновляем pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# 2. Устанавливаем PyTorch (ОДИН РАЗ).
# Версий 2.8/2.9 нет, ставим актуальную стабильную CPU-версию.
# Это самая тяжелая часть, делаем её в начале для кэша.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Копируем и устанавливаем остальные зависимости
COPY requirements.txt .

# ВАЖНО: Если в requirements.txt есть строчка 'torch', она может сломать сборку.
# Эта команда попытается установить пакеты, не обновляя уже установленный torch.
RUN pip install --no-cache-dir -r requirements.txt

# 4. Настройка окружения HuggingFace
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p $HF_HOME

# 5. Предзагружаем модель (Тест импорта torch и скачивание весов)
# Если здесь упадет — значит проблема в нехватке памяти (см. совет про .wslconfig выше)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    from transformers import AutoTokenizer, AutoModel; \
    model_id='BAAI/bge-m3'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModel.from_pretrained(model_id)"

# 6. Установка вашего локального модуля
COPY ms_graphrag_neo4j ./ms_graphrag_neo4j
RUN pip install -e ./ms_graphrag_neo4j

# 7. Копируем остальной код проекта
COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag:rag", "--host", "0.0.0.0", "--port", "8000"]