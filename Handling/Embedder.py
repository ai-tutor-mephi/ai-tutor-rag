from transformers import AutoTokenizer, AutoModel
import torch

import logging
from pathlib import Path

# путь к директории с текущим файлом
base_dir = Path(__file__).resolve().parent

# подняться на n директорий вверх
root_dir = base_dir.parents[1]

# путь к Logs
logs_dir = root_dir / "Logs"
logs_dir.mkdir(parents=True, exist_ok=True)  # создаём папку, если её нет

# сам лог-файл
log_file = logs_dir / "Embedder.log"

logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    filemode="a", 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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
    