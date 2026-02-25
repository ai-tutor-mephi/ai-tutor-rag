from transformers import AutoTokenizer, AutoModel
import torch

import logging
import sys
from utils.MyLogs import setup_logger

# Настройка логов
setup_logger(__file__)

model_id = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:

    def __init__(self, model: AutoModel.from_pretrained = model, tokenizer: AutoTokenizer.from_pretrained = tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    
    def embed(self, text: str) -> list[float]:
        """
        Embedding text. Получаем dense_vector
        :param text:
        :return:
        """
        logging.info("Векторизация текста...")
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model(**input).last_hidden_state

        embedding = embedding.mean(dim=1).squeeze(0)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)  # l2 norm
        embedding = embedding.tolist()  # list[float]

        logging.info(f"Текст векторизован.\n{embedding[:5]}... (всего {len(embedding)} чисел)")
        return embedding
    