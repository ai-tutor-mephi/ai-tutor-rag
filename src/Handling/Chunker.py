"""
Модуль для разбиения текста на чанки.

Этот модуль используется на этапе загрузки документов для разбиения
больших текстов на более мелкие фрагменты (чанки), которые затем
векторизуются и сохраняются в базу данных.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
import logging
from utils.MyLogs import setup_logger

# Настройка логов
setup_logger(__file__)


class Chunker:
    """
    Класс для разбиения текста на чанки.
    
    Использует рекурсивное разбиение текста с перекрытием между чанками,
    что позволяет сохранить контекст на границах фрагментов.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Инициализация разбивателя текста.
        
        Args:
            chunk_size: Максимальный размер чанка в символах
            chunk_overlap: Размер перекрытия между соседними чанками
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Инициализация разбивателя с приоритетными разделителями
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", "", "? ", "! "],
                                                       chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    

    def make_chunks_from_text(self, text: str) -> list[str] | None:
        try:
            logging.info("Создание чанков из текста...")

            chunks = self.splitter.split_text(text)

            logging.info(f"Чанки успешно созданы. Количество чанков: {len(chunks)}")
            return chunks

        except Exception as e:
            logging.error(f"Не удалось создать чанки: {e}")
            return []
