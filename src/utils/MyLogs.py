
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

_lock = threading.RLock()
_configured = False


def _repo_root() -> Path:
    """Каталог репозитория (есть pyproject.toml или rag.py)."""
    here = Path(__file__).resolve()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").is_file() or (d / "rag.py").is_file():
            return d
    return here.parents[2]


def configure_logging() -> None:
    """
    Идемпотентно настраивает root: один StreamHandler (stderr) и один FileHandler.

    Уровень: LOG_LEVEL (по умолчанию INFO). Каталог файлов: LOG_DIR или <repo>/Logs.
    """
    global _configured
    with _lock:
        if _configured:
            return
        _configured = True

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    log_dir = Path(os.getenv("LOG_DIR", str(_repo_root() / "Logs")))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except OSError as e:
        sys.stderr.write(f"[MyLogs] не удалось подключить файловый лог {log_dir}: {e}\n")

    http_level_name = os.getenv("LOG_LEVEL_HTTP", "WARNING").upper()
    http_level = getattr(logging, http_level_name, logging.WARNING)
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(http_level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True
