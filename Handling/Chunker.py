from langchain.text_splitter import RecursiveCharacterTextSplitter

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
log_file = logs_dir / "Chunker.log"

logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    filemode="a", 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", "", "? ", "! "],
                                                       chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
    

    def make_chunks_from_text(self, text: str) -> list[str] | None:
        try:
            logging.info("Создание чанков из текста...")

            chunks = self.splitter.split_text(text)

            logging.info(f"Чанки успешно созданы. Количество чанков: {len(chunks)}")
            return chunks

        except Exception as e:
            logging.error(f"Не удалось создать чанки: {e}")
            return []
