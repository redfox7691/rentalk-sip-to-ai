"""Utilities for extracting conversation logs and sending them via email."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional


CALL_MARKER = "=== NEW CALL ==="


def extract_last_call(log_path: str) -> Optional[str]:
    """Return the last call block from the conversation log."""

    if not log_path:
        return None

    path = Path(log_path)
    if not path.exists() or not path.is_file():
        return None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    if not lines:
        return None

    for idx in range(len(lines) - 1, -1, -1):
        if CALL_MARKER in lines[idx]:
            last_lines = lines[idx:]
            return "\n".join(last_lines) + "\n"

    return None


def send_conversation_email(subject: str, body: str) -> None:
    """Send the provided conversation body via SMTP email."""

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    mail_to = os.getenv("MAIL_TO")
    mail_from = os.getenv("MAIL_FROM") or smtp_user

    if not all([smtp_host, smtp_user, smtp_pass, mail_to]):
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = mail_from
    message["To"] = mail_to
    message.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(message)
