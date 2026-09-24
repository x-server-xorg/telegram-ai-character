"""Клиент ИИ для бота.

Три режима (DEEPSEEK_MODE):

* web     — «редирект веб-чата»: напрямую в chat.deepseek.com/api/v0/*
            (Bearer-токен из браузера + решение PoW-челленджа);
* official— официальный api.deepseek.com (OpenAI-совместимый REST);
* proxy   — любой OpenAI-совместимый прокси поверх веб-чата
            (deepseek2api, deepseek-free-api и подобные).
"""
from __future__ import annotations

import asyncio
import json
import logging

import aiohttp

import config
from pow import solve_pow, PowError

log = logging.getLogger("deepseek")

WEB_BASE = "https://chat.deepseek.com"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class AIError(RuntimeError):
    """Ошибка запроса к ИИ. retryable=False — повторять бессмысленно."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


# --------------------------------------------------------------------------
# разбор ответов
# --------------------------------------------------------------------------

def _parse_sse(body: str) -> tuple[str, dict]:
    """Собирает текст ответа из SSE-потока chat/completion.

    Реальный поток веб-чата DeepSeek (снят эмпирически):

      * ``event: ready``          — id запроса и ответа (цепочка сообщений);
      * ``event: update_session`` — снапшот сообщения ``{"v": {"response": ...}}``
        и дельты текста ``{"v": "<строка>"}`` / ``{"o": "APPEND", ...}``;
      * ``event: title``          — АВТОНАЗВАНИЕ беседы ``{"content": "..."}``
        (именно оно раньше выдавалось за ответ — игнорируется);
      * ``event: close``          — финальный чанк (игнорируется);
      * плюс служебные op-чанки ``SET``/``BATCH`` (статусы, токены) — игнорируются.

    Возвращает (текст ответа, служебные id).
    """
    text: list[str] = []
    ids: dict = {}
    openai_parts: list[str] = []
    event: str | None = None

    def walk(obj) -> None:
        if isinstance(obj, list):
            for item in obj:
                walk(item)
            return
        if not isinstance(obj, dict):
            return

        # формат OpenAI-подобного стрима (прокси)
        choices = obj.get("choices")
        if isinstance(choices, list):
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta") or ch.get("message") or {}
                if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"]:
                    openai_parts.append(delta["content"])
            return

        # автоназвание беседы — НЕ ответ модели
        if event in {"title", "close"}:
            return
        if event == "error":
            err = obj.get("content") or obj.get("message") or json.dumps(obj, ensure_ascii=False)[:200]
            raise AIError(f"DeepSeek: ошибка потока — {err}")

        for key in ("request_message_id", "response_message_id"):
            if obj.get(key) is not None:
                ids[key] = obj[key]

        v = obj.get("v")
        if isinstance(v, dict):
            resp = v.get("response")
            if isinstance(resp, dict):
                # снапшот сообщения — эталонный текст на данный момент
                frags = resp.get("fragments") or []
                snap = "".join(
                    (f.get("content") or "")
                    for f in frags
                    if isinstance(f, dict) and f.get("type") == "RESPONSE"
                )
                if snap:
                    text.clear()
                    text.append(snap)
                if resp.get("message_id") is not None:
                    ids["response_message_id"] = resp["message_id"]
            inner = v.get("data")
            if isinstance(inner, (dict, list)):
                walk(inner)
            return
        if isinstance(v, list):
            walk(v)
            return
        if isinstance(v, str):
            op = obj.get("o")
            # текстовая дельта — либо чисто {"v": "<текст>"}, либо APPEND-чанк;
            # внутри BATCH/SET лежат значения полей ({"p": ..., "v": ...}) — не текст
            if op == "APPEND" or set(obj.keys()) == {"v"}:
                text.append(v)
            return

        # голый content принимаем только у «текстовых» событий, не у metadata
        content = obj.get("content")
        if isinstance(content, str) and content and event in {None, "message", "delta"}:
            text.append(content)

        inner = obj.get("data")
        if isinstance(inner, (dict, list)):
            walk(inner)

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            event = line[6:].strip().lower()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            # не-JSON куски текстом не берём — так в ответ не попадёт мусор
            continue
        walk(obj)

    merged = "".join(text).strip() or "".join(openai_parts).strip()
    return merged, ids


# --------------------------------------------------------------------------
# клиенты
# --------------------------------------------------------------------------

class WebChatClient:
    """Прямое подключение к веб-чату DeepSeek (режим «редиректа»)."""

    name = "deepseek-web"

    def __init__(self) -> None:
        if not config.DEEPSEEK_TOKEN:
            raise AIError("DEEPSEEK_TOKEN не задан", retryable=False)
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.AI_TIMEOUT)
            )
        return self._session

    def _headers(self, pow_resp: str | None = None) -> dict:
        h = {
            "accept": "text/event-stream, application/json",
            "accept-language": "ru-RU,ru;q=0.9,en;q=0.8",
            "authorization": f"Bearer {config.DEEPSEEK_TOKEN}",
            "content-type": "application/json",
            "origin": WEB_BASE,
            "referer": f"{WEB_BASE}/",
            "user-agent": _UA,
            "x-app-version": "20241129.1",
            "x-client-locale": "ru_RU",
            "x-client-platform": "web",
            "x-client-version": "1.7.0",
        }
        if config.DEEPSEEK_COOKIE:
            h["cookie"] = config.DEEPSEEK_COOKIE
        if pow_resp:
            h["x-ds-pow-response"] = pow_resp
        return h

    async def _post(self, path: str, payload: dict | None, pow_resp: str | None = None):
        session = await self._sess()
        url = WEB_BASE + path
        try:
            async with session.post(url, json=payload or {}, headers=self._headers(pow_resp)) as resp:
                body = await resp.text()
                if resp.status == 401:
                    raise AIError(
                        "DeepSeek: токен недействителен или истёк (401). "
                        "Обновите DEEPSEEK_TOKEN в .env",
                        retryable=False,
                    )
                if resp.status == 403:
                    raise AIError("DeepSeek: доступ запрещён (403) — проверьте токен/cookie", retryable=False)
                if resp.status == 429:
                    raise AIError("DeepSeek: превышен лимит запросов (429)", retryable=True)
                if resp.status >= 500:
                    raise AIError(f"DeepSeek: серверная ошибка {resp.status}", retryable=True)
                if resp.status != 200:
                    raise AIError(f"DeepSeek: HTTP {resp.status}: {body[:300]}", retryable=True)
                return body
        except aiohttp.ClientError as exc:
            raise AIError(f"сетевая ошибка к chat.deepseek.com: {exc}", retryable=True) from exc

    @staticmethod
    def _check_code(body: str, what: str) -> dict:
        """Проверяет поле code у ответов chat.deepseek.com; возвращает data."""
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise AIError(f"DeepSeek: {what} — не-JSON ответ: {body[:300]}") from exc
        code = data.get("code")
        if code not in (0, None):
            msg = data.get("msg") or "неизвестная ошибка"
            low = str(msg).lower()
            authish = "token" in low or "auth" in low or code in (40001, 40002, 40003)
            raise AIError(
                f"DeepSeek: {what} — {msg} (code {code})"
                + (" — проверьте DEEPSEEK_TOKEN" if authish else ""),
                retryable=not authish,
            )
        return data.get("data") or {}

    async def _pow(self) -> str:
        body = await self._post(
            "/api/v0/chat/create_pow_challenge",
            {"target_path": "/api/v0/chat/completion"},
        )
        data = self._check_code(body, "PoW-челлендж")
        try:
            challenge = data["biz_data"]["challenge"]
        except (KeyError, TypeError) as exc:
            raise AIError(f"неожиданный ответ PoW-челленджа: {body[:300]}") from exc
        # CPU-задача — выносим из event loop
        try:
            return await asyncio.to_thread(solve_pow, challenge)
        except PowError as exc:
            raise AIError(f"PoW: {exc}", retryable=True) from exc

    async def _create_session(self) -> str:
        body = await self._post("/api/v0/chat_session/create", {})
        data = self._check_code(body, "создание сессии")
        try:
            sid = data["biz_data"]["id"]
        except (KeyError, TypeError) as exc:
            raise AIError(f"не удалось создать сессию DeepSeek: {body[:300]}") from exc
        return sid

    async def _decode_completion(self, body: str) -> tuple[str, dict]:
        """Разбирает тело chat/completion: SSE-поток либо JSON-ошибку."""
        if body.lstrip().startswith("{"):
            # не поток, а обычный JSON — обычно это ошибка (code != 0)
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            if isinstance(data, dict) and "code" in data:
                # здесь же всплывут ошибки вида «сессия не найдена»
                self._check_code(body, "completion")
        text, ids = _parse_sse(body)
        if not text:
            raise AIError(f"DeepSeek вернул пустой ответ: {body[:300]}")
        return text, ids

    @staticmethod
    def _session_error(exc: AIError) -> bool:
        low = str(exc).lower()
        return "session" in low or "сесс" in low

    async def complete(
        self,
        system: str,
        user: str,
        *,
        session_id: str | None = None,
        parent_id: int | None = None,
    ) -> tuple[str, dict]:
        """Ответ ИИ. Возвращает (текст, meta), где meta — id сессии/сообщений.

        ``session_id``/``parent_id`` — постоянный чат DeepSeek: сессия
        переиспользуется и цепочка сообщений продолжается (parent_message_id),
        пока пользователь не сделает /forget. Если сессия устарела или удалена —
        создаётся новая, запрос повторяется один раз.
        """
        prompt = (
            "[СИСТЕМНАЯ ИНСТРУКЦИЯ]\n"
            f"{system.strip()}\n\n"
            "[ЗАДАНИЕ]\n"
            f"{user.strip()}"
        )
        pow_resp = await self._pow()
        sid = session_id
        may_recreate = sid is not None  # разрешено одно пересоздание сессии

        while True:
            if not sid:
                sid = await self._create_session()
            payload = {
                "chat_session_id": sid,
                "parent_message_id": parent_id,
                "prompt": prompt,
                "ref_file_ids": [],
                "thinking_enabled": config.DEEPSEEK_THINKING,
                "search_enabled": False,
            }
            try:
                body = await self._post("/api/v0/chat/completion", payload, pow_resp)
                text, ids = await self._decode_completion(body)
                break
            except AIError as exc:
                if may_recreate and exc.retryable and self._session_error(exc):
                    log.warning("сессия DeepSeek недействительна (%s) — пересоздаю", exc)
                    may_recreate = False
                    sid = None
                    continue
                raise

        meta = {
            "session_id": sid,
            "response_message_id": ids.get("response_message_id"),
            "request_message_id": ids.get("request_message_id"),
        }
        return text, meta

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class OpenAICompatClient:
    """OpenAI-совместимый REST (официальный API или локальный прокси)."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.AI_TIMEOUT)
            )
        return self._session

    async def complete(
        self,
        system: str,
        user: str,
        *,
        session_id: str | None = None,
        parent_id: int | None = None,
    ) -> tuple[str, dict]:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        # deepseek-reasoner не принимает temperature
        if "reasoner" not in self.model:
            payload["temperature"] = config.TEMPERATURE

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"

        session = await self._sess()
        url = f"{self.base_url}/chat/completions"
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.text()
                if resp.status == 401:
                    raise AIError(f"{self.name}: неверный API-ключ (401)", retryable=False)
                if resp.status == 429:
                    raise AIError(f"{self.name}: лимит запросов (429)")
                if resp.status != 200:
                    raise AIError(f"{self.name}: HTTP {resp.status}: {body[:300]}")
        except aiohttp.ClientError as exc:
            raise AIError(f"сетевая ошибка {self.name}: {exc}") from exc

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AIError(f"{self.name}: неожиданный формат ответа: {body[:300]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise AIError(f"{self.name}: пустой ответ")
        return content.strip(), {}

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


def make_client():
    mode = config.DEEPSEEK_MODE
    if mode == "web":
        return WebChatClient()
    if mode == "official":
        return OpenAICompatClient(
            name="deepseek-official",
            base_url="https://api.deepseek.com/v1",
            api_key=config.DEEPSEEK_API_KEY,
            model=config.DEEPSEEK_MODEL or "deepseek-chat",
        )
    if mode == "proxy":
        return OpenAICompatClient(
            name="deepseek-proxy",
            base_url=config.DEEPSEEK_BASE_URL,
            api_key=config.DEEPSEEK_API_KEY or "none",
            model=config.DEEPSEEK_MODEL or "deepseek-chat",
        )
    raise AIError(f"неизвестный DEEPSEEK_MODE={mode!r}", retryable=False)


async def complete_retry(
    client,
    system: str,
    user: str,
    *,
    session_id: str | None = None,
    parent_id: int | None = None,
    attempts: int = 3,
) -> tuple[str, dict]:
    """Запрос к ИИ с повторами на временные ошибки. Возвращает (текст, meta)."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await client.complete(
                system, user, session_id=session_id, parent_id=parent_id
            )
        except AIError as exc:
            last = exc
            if not exc.retryable or attempt == attempts - 1:
                raise
            log.warning("ИИ (попытка %d): %s", attempt + 1, exc)
        except asyncio.TimeoutError as exc:
            last = exc
            if attempt == attempts - 1:
                raise AIError("ИИ: таймаут запроса") from exc
            log.warning("ИИ: таймаут (попытка %d)", attempt + 1)
        await asyncio.sleep(delay)
        delay *= 2
    raise AIError(f"ИИ: исчерпаны попытки: {last}")
