from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

_lock = threading.RLock()
_done = False


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "pyproject.toml").is_file() or (d / "rag.py").is_file():
            return d
    return here


def configure_logging() -> None:
    global _done
    with _lock:
        if _done:
            return
        _done = True

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

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
        sys.stderr.write(f"[logging_setup] не удалось создать файловый лог {log_dir}: {e}\n")

    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(
            getattr(logging, os.getenv("LOG_LEVEL_HTTP", "WARNING").upper(), logging.WARNING)
        )
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True
