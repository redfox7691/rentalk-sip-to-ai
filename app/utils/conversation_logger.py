"""Utility for persisting human/agent utterances to a shared log file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import structlog

from app.config import config


class ConversationLogger:
    """Append plain-text conversation entries to a configured log file."""

    def __init__(self, log_path: Optional[str]) -> None:
        self._logger = structlog.get_logger(__name__)
        self.log_path = None
        self._lock = Lock()

        if not log_path:
            self._logger.info("Conversation logging disabled - no CONVERSATION_LOG_PATH provided")
            return

        try:
            path = Path(log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            self.log_path = path
            self._logger.info("Conversation logging enabled", path=str(path))
        except Exception as exc:
            self._logger.error(
                "Unable to initialize conversation logger",
                path=log_path,
                error=str(exc)
            )
            self.log_path = None

    def log_user(self, text: Optional[str]) -> None:
        """Record a user utterance."""
        self._write_entry("USER", text)

    def log_agent(self, text: Optional[str]) -> None:
        """Record an AI/agent utterance."""
        self._write_entry("AGENT", text)

    def _write_entry(self, role: str, text: Optional[str]) -> None:
        if not self.log_path or not text:
            return

        cleaned = text.strip()
        if not cleaned:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        line = f"{timestamp} {role}: {cleaned}\n"

        try:
            with self._lock:
                with self.log_path.open("a", encoding="utf-8") as fp:
                    fp.write(line)
        except Exception as exc:
            self._logger.error(
                "Failed to write conversation log entry",
                path=str(self.log_path),
                error=str(exc)
            )


conversation_logger = ConversationLogger(config.system.conversation_log_path)

__all__ = ["ConversationLogger", "conversation_logger"]
