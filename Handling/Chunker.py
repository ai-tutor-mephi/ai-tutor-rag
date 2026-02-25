from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
import logging
from utils.MyLogs import setup_logger

# Настройка логов
setup_logger(__file__)

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
