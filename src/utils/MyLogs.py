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
    
    # Создаем директорию для логов относительно корня проекта
    # Определяем корень проекта: ищем директорию с utils/ или поднимаемся до корня
    file_path_obj = Path(file_path).resolve()
    current = file_path_obj.parent
    
    # Ищем корень проекта (где находится utils/ или другие характерные файлы)
    # Поднимаемся вверх, пока не найдем директорию с utils/
    project_root = current
    while project_root != project_root.parent:
        if (project_root / "utils").exists() or (project_root / "requirements.txt").exists():
            break
        project_root = project_root.parent
    
    # Если не нашли, используем директорию на 2 уровня выше от файла
    if project_root == project_root.parent:
        project_root = file_path_obj.parent.parent
    
    logs_dir = project_root / "Logs"
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
