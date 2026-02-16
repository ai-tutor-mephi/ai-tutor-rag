FROM python:3.12-slim

WORKDIR /app


# Устанавливаем pip зависимости в одном слое для лучшего кэширования
# Сначала устанавливаем PyTorch, затем остальные зависимости
RUN pip install --upgrade pip && \
    pip install torch==2.8.0+cpu torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Копируем и устанавливаем зависимости из requirements.txt
# Это делается до копирования кода для лучшего кэширования
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем и устанавливаем ms_graphrag_neo4j до копирования основного кода
# Это позволяет кэшировать установку пакета отдельно от изменений кода
COPY ms_graphrag_neo4j ./ms_graphrag_neo4j
RUN pip install -e ./ms_graphrag_neo4j

# Создаем директорию для моделей (модель будет загружена при первом запуске)
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p $HF_HOME

# Копируем код приложения в последнюю очередь
# Это позволяет максимально использовать кэш для зависимостей
COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag:rag", "--host", "0.0.0.0", "--port", "8000"]

