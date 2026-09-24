"""Telegram-бот: ИИ-персонаж с постоянной памятью диалога (DeepSeek).

Характер — в init.txt. Каждый чат живёт в ОДНОЙ постоянной сессии DeepSeek
(создаётся один раз, переиспользуется до /forget) — история хранится на сервере,
как в character.ai. Локально пишется только архив messages.jsonl.

Запуск:  python bot.py
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
import time

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message

import config
from brain import clear_memory, get_memory, init_hash, load_session, save_session
from deepseek import AIError, complete_retry, make_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bot")

dp = Dispatcher()
AI = None  # клиент ИИ создаётся в main() после проверки конфигурации

BOT_ID: int | None = None
BOT_USERNAME: str = ""
BOT_TITLE: str = ""

_reply_locks: dict[int, asyncio.Lock] = {}

TG_LIMIT = 3900

HELP = (
    "Я — ИИ-персонаж. Характер задаётся файлом init.txt, а память устроена так: "
    "каждый наш чат — один постоянный разговор, я помню всё с самого начала "
    "и не теряю нить даже после перерывов.\n\n"
    "Команды:\n"
    "/forget — очистить память диалога в этом чате (и начать новый разговор)\n"
    "/help — эта справка"
)


# --------------------------------------------------------------------------
# утилиты
# --------------------------------------------------------------------------

def message_text(message: Message) -> str | None:
    """Текст сообщения любого типа (или понятная заглушка)."""
    if message.text:
        return message.text
    if message.caption:
        return message.caption
    if message.sticker:
        emoji = f" {message.sticker.emoji}" if message.sticker.emoji else ""
        return f"[стикер{emoji}]"
    if message.photo:
        return "[фото]"
    if message.video:
        return "[видео]"
    if message.voice:
        return "[голосовое]"
    if message.audio:
        return "[аудио]"
    if message.video_note:
        return "[кружок]"
    if message.document:
        return "[документ]"
    if message.contact:
        return "[контакт]"
    if message.location or message.venue:
        return "[геолокация]"
    if message.poll:
        return "[опрос]"
    if message.story:
        return "[история]"
    return None


def author_name(message: Message) -> str:
    user = message.from_user
    if user is None:
        return message.chat.title or "аноним"
    return user.full_name or user.username or str(user.id)


def should_reply(message: Message, text: str) -> bool:
    """Решает, отвечать ли на сообщение (записывать в память бот обязан всегда)."""
    if message.chat.type == "private":
        return True
    low = text.lower()
    if f"@{BOT_USERNAME}".lower() in low:
        return True
    replied = message.reply_to_message
    if replied and replied.from_user and replied.from_user.id == BOT_ID:
        return True
    if text.startswith("/"):
        return True
    for word in config.TRIGGER_WORDS:
        if word in low:
            return True
    return False


def split_chunks(text: str, limit: int = TG_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    buf = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(buf) + len(line) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def is_admin(user_id: int | None) -> bool:
    if not config.ADMIN_IDS:
        return True
    return user_id in config.ADMIN_IDS


def reply_lock(chat_id: int) -> asyncio.Lock:
    lock = _reply_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _reply_locks[chat_id] = lock
    return lock


WEAK_LEN = 15  # меньше — точно односложно


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower(), flags=re.U).strip()


def is_weak_answer(answer: str, incoming: str) -> bool:
    """Односложный ответ или эхо реплики собеседника → стоит попросить развить."""
    a = (answer or "").strip()
    if len(a) < WEAK_LEN:
        return True
    na, ni = _norm(a), _norm(incoming or "")
    if not na or not ni:
        return False
    if na == ni or na in ni:
        return True  # ответ дословно повторяет (или целиком внутри) чужую реплику
    # эхо с парой лишних слов: цитата чужой реплики почти без добавления
    if len(ni) >= 6 and ni in na and len(na) - len(ni) <= 15:
        return True
    return False


# --------------------------------------------------------------------------
# команды
# --------------------------------------------------------------------------

@dp.message(Command("start", "help"))
async def cmd_start(message: Message):
    await message.answer(HELP)


@dp.message(Command("forget"))
async def cmd_forget(message: Message):
    if not is_admin(message.from_user.id if message.from_user else None):
        await message.answer("Команда доступна только админам.")
        return
    existed = clear_memory(message.chat.id)
    await message.answer(
        "Память диалога очищена — начинаю с чистого листа."
        if existed else
        "Память и так пуста."
    )


# --------------------------------------------------------------------------
# основной обработчик: принимаем ВСЕ сообщения
# --------------------------------------------------------------------------

@dp.message()
async def on_message(message: Message):
    text = message_text(message)
    if text is None:
        return  # служебные/пустые апдейты не пишем

    from_user = message.from_user
    chat_id = message.chat.id
    chat_type = message.chat.type
    title = message.chat.title or message.chat.first_name or str(chat_id)
    memory = get_memory(chat_id, chat_type, title)

    name = author_name(message)
    user_id = from_user.id if from_user else None

    # 1) всё пишется в память диалога
    memory.add(name, user_id, text, kind="user")

    # 2) других ботов не дразним
    if from_user and from_user.is_bot and from_user.id != BOT_ID:
        return

    # 3) отвечаем только когда обращаются
    if not should_reply(message, text):
        return

    async with reply_lock(chat_id):
        # постоянная сессия DeepSeek: создаётся один раз и живёт до /forget
        sess = load_session(chat_id)
        include_init = (
            not sess.get("session_id")
            or sess.get("init_hash") != init_hash()
        )
        try:
            system, user_prompt = memory.build_prompts(
                text, sender=name, include_init=include_init
            )
            answer, meta = await asyncio.wait_for(
                complete_retry(
                    AI, system, user_prompt,
                    session_id=sess.get("session_id"),
                    parent_id=sess.get("parent_id"),
                ),
                timeout=config.AI_TIMEOUT + 30,
            )

            # односложие/эхо → одна попытка получить живой развёрнутый ответ
            if is_weak_answer(answer, text):
                log.info("chat %s: односложный ответ, повтор с требованием развить", chat_id)
                try:
                    answer, meta = await asyncio.wait_for(
                        complete_retry(
                            AI, system,
                            user_prompt
                            + "\n\n[ВАЖНО] Ответь развёрнутее: полноценное предложение "
                              "с мыслью, не повторяй слова собеседника дословно.",
                            session_id=meta.get("session_id") or sess.get("session_id"),
                            parent_id=meta.get("response_message_id"),
                        ),
                        timeout=config.AI_TIMEOUT + 30,
                    )
                except (AIError, asyncio.TimeoutError) as exc:
                    log.info("chat %s: повтор не удался (%s), оставляю первый вариант", chat_id, exc)
        except AIError as exc:
            log.error("chat %s: ИИ недоступен: %s", chat_id, exc)
            await message.answer(f"ИИ сейчас недоступен: {exc}")
            return
        except asyncio.TimeoutError:
            log.error("chat %s: таймаут ИИ", chat_id)
            await message.answer("ИИ не успел ответить (таймаут), попробуй ещё раз.")
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("chat %s: непредвиденная ошибка", chat_id)
            await message.answer(f"Ошибка: {exc}")
            return

        # запоминаем сессию и цепочку сообщений — следующий ответ продолжит её
        if meta.get("session_id"):
            save_session(chat_id, {
                "session_id": meta.get("session_id"),
                "parent_id": meta.get("response_message_id"),
                "init_hash": init_hash(),
                "updated": int(time.time()),
            })

        answer = answer.strip()[: config.MAX_REPLY_CHARS]
        # свой ответ тоже пишем в память — бот помнит, что сам сказал
        memory.add(BOT_TITLE or "бот", BOT_ID, answer, kind="assistant")

        for chunk in split_chunks(answer):
            await message.answer(chunk)


# --------------------------------------------------------------------------
# точка входа
# --------------------------------------------------------------------------

async def main() -> None:
    errors = config.validate()
    if errors:
        for err in errors:
            log.error("Конфигурация: %s", err)
        log.error("Исправьте .env (см. .env.example) и запустите снова.")
        sys.exit(1)

    global AI
    AI = make_client()

    bot = Bot(
        token=config.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=None),
    )
    me = await bot.get_me()
    global BOT_ID, BOT_USERNAME, BOT_TITLE
    BOT_ID = me.id
    BOT_USERNAME = me.username or ""
    BOT_TITLE = me.first_name or BOT_USERNAME

    log.info("Бот запущен: @%s (%s)", BOT_USERNAME, BOT_TITLE)
    log.info("Режим DeepSeek: %s | init.txt: %s | данные: %s",
             config.DEEPSEEK_MODE, config.INIT_FILE, config.DATA_DIR)

    try:
        await dp.start_polling(bot)
    finally:
        close = getattr(AI, "aclose", None)
        if close:
            await close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Остановлено.")
