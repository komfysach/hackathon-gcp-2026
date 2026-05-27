"""Per-agent per-round trace logging using contextvars for async isolation."""
from __future__ import annotations

import contextvars
import logging
from pathlib import Path

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("_trace_id", default="")
_LOG_NAMES = ("google_adk", "google_genai")
_log_handlers: dict[str, logging.FileHandler] = {}


class _TraceFilter(logging.Filter):
    def __init__(self, tag: str):
        super().__init__()
        self._tag = tag

    def filter(self, record: logging.LogRecord) -> bool:
        return _trace_id.get("") == self._tag


def start_trace(logs_dir: Path, spec_name: str, round_id: int) -> Path:
    logs_dir.mkdir(exist_ok=True)
    tag = f"{spec_name}_{round_id}"
    _trace_id.set(tag)
    log_path = logs_dir / f"{spec_name}_round_{round_id}.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s"))
    handler.addFilter(_TraceFilter(tag))
    _log_handlers[tag] = handler
    for name in _LOG_NAMES:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
    return log_path


def stop_trace(spec_name: str, round_id: int) -> None:
    tag = f"{spec_name}_{round_id}"
    handler = _log_handlers.pop(tag, None)
    if handler:
        for name in _LOG_NAMES:
            logging.getLogger(name).removeHandler(handler)
        handler.close()
