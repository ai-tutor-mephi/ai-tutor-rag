import logging
import sys
from pathlib import Path

def setup_logger(file_path: str):
    """
    Инициализирует логи для файла, автоматически определяя имя файла из пути.
    
    Args:
        file_path: Путь к файлу (обычно передается __file__)
    
    Returns:
        Настроенный logger
    """
    # Определяем имя файла без расширения из пути
    file_name = Path(file_path).stem
    
    # Создаем директорию для логов
    logs_dir = Path("/Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{file_name}.log"
    
    # Настраиваем root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Перезаписываем существующую конфигурацию
    )
    
    return logging.getLogger(file_name)
