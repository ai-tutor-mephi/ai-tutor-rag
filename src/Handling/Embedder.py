"""
Модуль для векторизации текста (embedding).

Этот модуль преобразует текстовые данные в векторные представления,
которые используются для семантического поиска в векторной БД.
Использует модель BAAI/bge-m3 для создания embeddings.
"""

from transformers import AutoTokenizer, AutoModel
import torch

import logging

logger = logging.getLogger(__name__)

# Инициализация модели для векторизации
model_id = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """
    Класс для векторизации текста.
    
    Преобразует текстовые строки в плотные векторы (dense vectors),
    которые используются для семантического поиска в Qdrant.
    """

    def __init__(self, model: AutoModel.from_pretrained = model, tokenizer: AutoTokenizer.from_pretrained = tokenizer):
        """
        Инициализация векторизатора.
        
        Args:
            model: Модель для создания embeddings
            tokenizer: Токенизатор для обработки текста
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()  # Переводим модель в режим инференса

    
    def embed(self, text: str) -> list[float]:
        """
        Преобразует текст в векторное представление (embedding).
        
        Процесс:
        1. Токенизация текста
        2. Получение embeddings через модель
        3. Усреднение по токенам
        4. Нормализация (L2 norm)
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Список чисел (вектор) размерностью модели
        """
        logger.info("Векторизация текста...")
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model(**input).last_hidden_state

        embedding = embedding.mean(dim=1).squeeze(0)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)  # l2 norm
        embedding = embedding.tolist()  # list[float]

        logger.info(
            "Текст векторизован. Первые 5 значений: %s... (всего %s чисел)",
            embedding[:5],
            len(embedding),
        )
        return embedding
    