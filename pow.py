"""Решатель Proof-of-Work веб-чата DeepSeek (алгоритм DeepSeekHashV1).

Стратегия:
1) основной путь — официальный WASM-солвер DeepSeek (``vendor/deepseek_pow.wasm``),
   который выполняется через Node.js (``vendor/pow_solver.js``). Именно так
   работают проверенные обёртки над веб-чатом, и только этот путь гарантированно
   принимается сервером.
2) резерв — «чистый» Python-перебор по формуле из открытых описаний:
   Keccak-256(f"{salt}_{expire_at}_{nonce}") == challenge.
   Формула из документации сообщества на практике может не совпасть с
   алгоритмом сервера, поэтому фолбэк помечен как экспериментальный.

Ответ упаковывается в JSON, кодируется в base64 и уходит в заголовке
``x-ds-pow-response``.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

try:  # быстрый путь (C-реализация)
    from Crypto.Hash import keccak as _keccak_mod

    def keccak256_hex(data: bytes) -> str:
        return _keccak_mod.new(digest_bits=256, data=data).hexdigest()

    _HAS_PYCRYPTODOME = True
except ImportError:  # медленный, но автономный fallback
    _HAS_PYCRYPTODOME = False

    def keccak256_hex(data: bytes) -> str:  # type: ignore[misc]
        return _pure_keccak256(data).hex()


# --- чистый Keccak-256 (паддинг 0x01, как у Ethereum/DeepSeek, НЕ SHA3-256) ---

_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

_ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]


def _rol(x: int, n: int) -> int:
    n %= 64
    return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF


def _keccak_f(st: list[list[int]]) -> None:
    for rnd in range(24):
        c = [st[x][0] ^ st[x][1] ^ st[x][2] ^ st[x][3] ^ st[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                st[x][y] ^= d[x]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rol(st[x][y], _ROT[x][y])
        for x in range(5):
            for y in range(5):
                st[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
        st[0][0] ^= _RC[rnd]


def _pure_keccak256(data: bytes) -> bytes:
    rate = 136  # 1088 бит для Keccak-256
    st = [[0] * 5 for _ in range(5)]

    padded = bytearray(data)
    pad_len = rate - (len(padded) % rate)
    padded += b"\x01" + b"\x00" * (pad_len - 2) + b"\x80" if pad_len >= 2 else b"\x81"

    for off in range(0, len(padded), rate):
        block = padded[off:off + rate]
        for i in range(rate // 8):
            st[i % 5][i // 5] ^= int.from_bytes(block[i * 8:(i + 1) * 8], "little")
        _keccak_f(st)

    out = bytearray()
    for i in range(4):  # 32 байта = 4 лanes
        out += st[i % 5][i // 5].to_bytes(8, "little")
    return bytes(out)


class PowError(RuntimeError):
    pass


_WASM_CLI = Path(__file__).resolve().parent / "vendor" / "pow_solver.js"


# --------------------------------------------------------------------------
# основной путь: официальный WASM через Node.js
# --------------------------------------------------------------------------

def _solve_pow_wasm(challenge: dict, timeout: float = 60.0) -> str:
    node = shutil.which("node")
    if not node:
        raise PowError("Node.js не найден в PATH (нужен для WASM-солвера PoW)")

    proc = subprocess.run(
        [node, str(_WASM_CLI)],
        input=json.dumps(challenge, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[:300]
        raise PowError(f"WASM-солвер завершился с ошибкой: {err}")

    header = proc.stdout.decode("ascii", "replace").strip()
    if not header:
        raise PowError("WASM-солвер вернул пустой ответ")
    # проверяем, что это валидный payload с answer
    try:
        payload = json.loads(base64.b64decode(header + "=" * (-len(header) % 4)))
        int(payload["answer"])
    except Exception as exc:  # noqa: BLE001
        raise PowError(f"некорректный payload от WASM-солвера: {header[:80]}") from exc
    return header


# --------------------------------------------------------------------------
# резервный путь: «чистый» Python (экспериментальный)
# --------------------------------------------------------------------------

def _solve_pow_python(challenge: dict) -> str:
    """Перебор по формуле Keccak-256(f"{salt}_{expire_at}_{nonce}") == challenge.

    ВНИМАНИЕ: формула взята из открытых описаний и на текущих челленджах
    DeepSeek может не совпадать с алгоритмом сервера — используйте только когда
    Node.js недоступен, и проверяйте результат.
    """
    algorithm = challenge.get("algorithm")
    if algorithm != "DeepSeekHashV1":
        raise PowError(f"неподдерживаемый алгоритм PoW: {algorithm!r}")

    salt = str(challenge["salt"])
    expire_at = challenge["expire_at"]
    difficulty = int(challenge.get("difficulty") or 144000)
    target = str(challenge["challenge"]).strip().lower()

    prefix = f"{salt}_{expire_at}_".encode()
    answer: int | None = None
    for nonce in range(difficulty + 1):
        if keccak256_hex(prefix + str(nonce).encode()) == target:
            answer = nonce
            break
    if answer is None:
        raise PowError(
            f"python-фолбэк: nonce не найден за {difficulty} итераций "
            f"(формула сообщества не совпала с сервером — нужен Node.js/WASM)"
        )
    return _encode_payload(challenge, answer)


def _encode_payload(challenge: dict, answer: int) -> str:
    payload = {
        "algorithm": challenge.get("algorithm"),
        "challenge": challenge.get("challenge"),
        "salt": str(challenge.get("salt")),
        "answer": answer,
        "signature": str(challenge.get("signature", "")),
        "target_path": str(challenge.get("target_path", "")),
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(blob).decode("ascii")


def solve_pow(challenge: dict) -> str:
    """Возвращает значение заголовка x-ds-pow-response."""
    # 1) официальный WASM (нужен node)
    if _WASM_CLI.exists() and shutil.which("node"):
        try:
            return _solve_pow_wasm(challenge)
        except PowError:
            raise
        except Exception as exc:  # noqa: BLE001 — падаем в python-фолбэк
            log_warning = f"WASM-солвер недоступен ({exc}), пробую python-фолбэк"
            import sys
            print(f"[pow] {log_warning}", file=sys.stderr)
    # 2) python-фолбэк
    return _solve_pow_python(challenge)


if __name__ == "__main__":
    # самопроверка: хеши совпадают у pycryptodome и pure-реализации
    assert keccak256_hex(b"") == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert keccak256_hex(b"abc") == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
    print("pow.py: Keccak-256 ok (pycryptodome=%s)" % _HAS_PYCRYPTODOME)
    print("pow.py: WASM-CLI:", "найден" if _WASM_CLI.exists() else "НЕ найден",
          "| node:", shutil.which("node") or "НЕ найден")
