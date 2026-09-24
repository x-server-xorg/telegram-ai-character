"""Конфигурация бота: переменные из .env + значения по умолчанию."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _ids(name: str) -> set[int]:
    out: set[int] = set()
    for part in os.getenv(name, "").split(","):
        part = part.strip()
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            out.add(int(part))
    return out


# --- Telegram ---
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_IDS: set[int] = _ids("ADMIN_IDS")

# --- DeepSeek ---
DEEPSEEK_MODE: str = os.getenv("DEEPSEEK_MODE", "web").strip().lower()
DEEPSEEK_TOKEN: str = os.getenv("DEEPSEEK_TOKEN", "").strip()
DEEPSEEK_COOKIE: str = os.getenv("DEEPSEEK_COOKIE", "").strip()
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL: str = os.getenv(
    "DEEPSEEK_BASE_URL", "http://127.0.0.1:8000/v1"
).strip()
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_THINKING: bool = os.getenv("DEEPSEEK_THINKING", "0").strip() in {"1", "true", "yes", "on"}

# --- Память диалога ---
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).expanduser()
INIT_FILE: Path = Path(os.getenv("INIT_FILE", str(BASE_DIR / "init.txt"))).expanduser()

# Единственный предел контекста — размер транскрипта в промпте.
# Это физическое ограничение контекстного окна модели; стоит его глубоко,
# чтобы диалог держался максимально долго (обрезаются только самые старые реплики).
MAX_CONTEXT_CHARS: int = max(500, _int("MAX_CONTEXT_CHARS", 100000))
MAX_REPLY_CHARS: int = max(200, _int("MAX_REPLY_CHARS", 3500))
AI_TIMEOUT: int = max(10, _int("AI_TIMEOUT", 180))
TEMPERATURE: float = _float("TEMPERATURE", 0.9)
TRIGGER_WORDS: list[str] = [
    w.strip().lower()
    for w in os.getenv("TRIGGER_WORDS", "").split(",")
    if w.strip()
]

DATA_DIR.mkdir(parents=True, exist_ok=True)


def validate() -> list[str]:
    """Список фатальных проблем конфигурации (пусто = всё ок)."""
    errors: list[str] = []
    if not TELEGRAM_BOT_TOKEN:
        errors.append("не задан TELEGRAM_BOT_TOKEN (получите токен у @BotFather)")
    if DEEPSEEK_MODE not in {"web", "official", "proxy"}:
        errors.append(f"неизвестный DEEPSEEK_MODE={DEEPSEEK_MODE!r} (web|official|proxy)")
    if DEEPSEEK_MODE == "web" and not DEEPSEEK_TOKEN:
        errors.append(
            "DEEPSEEK_MODE=web, но не задан DEEPSEEK_TOKEN "
            "(chat.deepseek.com -> F12 -> Application -> Local Storage -> userToken)"
        )
    if DEEPSEEK_MODE == "official" and not DEEPSEEK_API_KEY:
        errors.append("DEEPSEEK_MODE=official, но не задан DEEPSEEK_API_KEY")
    return errors
