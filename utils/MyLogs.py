import logging
import sys
from pathlib import Path

def setup_logger(name: str, filename: str):
    logs_dir = Path("Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / filename


    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # запрещаем уход вверх

    # удаляем старые хендлеры, если уже есть
    if logger.hasHandlers():
        logger.handlers.clear()

    # создаём хендлеры
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)

    # задаём формат
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)

    # добавляем хендлеры
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
