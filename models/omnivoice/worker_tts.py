"""tts-worker: minimal GPU TTS HTTP server (stdlib only).

POST /tts {text, instruct?, seed?}
  -> omnivoice-tts (stdin -> /tmp WAV) -> ffmpeg -> MP3 (64k mono) -> bytes
GET /health -> {"ok": true}

One synthesis at a time (threading.Lock) — the GPU is the bottleneck and 8 GB
VRAM is only safe with concurrency 1. Concurrent requests block in HTTP.

Persian normalization + a small instruct whitelist, included inline with
no external dependencies.

Env: MODEL_DIR=/models  TTS_BIN=/opt/omnivoice.cpp/build/omnivoice-tts
     BITRATE=64k  TIMEOUT_S=120  PORT=8300  MAX_CHARS=1500
     ARCHIVE_DIR=/archive (optional: every successful synthesis saves
     MP3 + JSON sidecar here; empty/unset = archiving OFF; host dir comes
     from the compose volume; failures never break TTS)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
TTS_BIN = os.environ.get("TTS_BIN", "/opt/omnivoice.cpp/build/omnivoice-tts")
BITRATE = os.environ.get("BITRATE", "64k")
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "120"))
PORT = int(os.environ.get("PORT", "8300"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1500"))
LANG = os.environ.get("TTS_LANG", "Persian")

BASE_GGUF = os.path.join(MODEL_DIR, "tts/omnivoice-q4/omnivoice-base-Q4_K_M.gguf")
CODEC_GGUF = os.path.join(MODEL_DIR, "tts/omnivoice-q4/omnivoice-tokenizer-Q4_K_M.gguf")

# Arabic forms -> Persian; strip tashkeel. Conservative: ة left alone.
_AR_TO_FA = str.maketrans({"ي": "ی", "ك": "ک", "ﺔ": "ه"})
_TASHKEEL = re.compile("[\u064b-\u0652\u0670]")
_WS = re.compile(r"\s+")
_END_PUNCT = (".", "!", "?", "؟", "。", "！", "？", "…")

# The binary hard-rejects free-form text: fixed whitelist only (verified
# empirically 2026-09-07 — unknown items exit 1).
INSTRUCT_ITEMS = {
    "female", "male", "child", "teenager", "young adult", "middle-aged",
    "elderly", "high pitch", "moderate pitch", "low pitch",
    "very high pitch", "very low pitch", "whisper",
}

_GPU_LOCK = threading.Lock()

# Optional local audio archive (host dir via compose volume).
# Empty/unset = OFF. Anonymous filenames (no user identity - the worker only
# sees text); match sidecar text/chars/ts against your server access logs.
ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "").strip()
_TEHRAN = timezone(timedelta(hours=3, minutes=30))


def archive_save(mp3: bytes, *, text: str, voice: str | None, seed: int,
                 audio_s: float, gen_ms: int) -> str | None:
    """Save MP3 + JSON sidecar to ARCHIVE_DIR. Never raises (TTS must not
    break on archive failure). Returns mp3 path or None (off/failed)."""
    if not ARCHIVE_DIR:
        return None
    try:
        os.makedirs(ARCHIVE_DIR, mode=0o755, exist_ok=True)
        now = datetime.now(_TEHRAN)
        stamp = now.strftime("%Y%m%d-%H%M%S")
        safe_voice = re.sub(r"[^a-z0-9]+", "", (voice or "auto").lower())[:20] or "auto"
        base = "%s_%s_%dch_%s" % (stamp, safe_voice, len(text), uuid.uuid4().hex[:6])
        mp3_path = os.path.join(ARCHIVE_DIR, base + ".mp3")
        json_path = os.path.join(ARCHIVE_DIR, base + ".json")
        with open(mp3_path, "wb") as f:
            f.write(mp3)
        sidecar = {
            "ts": time.time(),
            "day_tehran": now.strftime("%Y-%m-%d"),
            "voice": voice or "auto",
            "seed": seed,
            "chars": len(text),
            "audio_s": audio_s,
            "gen_ms": gen_ms,
            "bitrate": BITRATE,
            "text": text[:1500],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        print("archive saved %s.mp3 (%d bytes)" % (base, len(mp3)), flush=True)
        return mp3_path
    except Exception as exc:
        print("archive skip: %s: %s" % (type(exc).__name__, exc), flush=True)
        return None


def normalize_persian(text: str) -> str:
    cleaned = _WS.sub(" ", text.translate(_AR_TO_FA))
    cleaned = _TASHKEEL.sub("", cleaned).strip()
    if not cleaned:
        raise ValueError("text is empty after normalization")
    if len(cleaned) > MAX_CHARS:
        raise ValueError(f"text is {len(cleaned)} chars, limit is {MAX_CHARS}")
    if not cleaned.endswith(_END_PUNCT):
        cleaned += "."
    return cleaned


def validate_instruct(raw: object) -> str | None:
    if raw is None or not str(raw).strip():
        return None
    value = " ".join(str(raw).split())
    if len(value) > 200:
        raise ValueError("instruct must be <= 200 chars")
    items = [i.strip().lower() for i in value.split(",") if i.strip()]
    bad = [i for i in items if i not in INSTRUCT_ITEMS]
    if bad:
        raise ValueError(
            f"unsupported instruct item(s): {bad}. valid: {sorted(INSTRUCT_ITEMS)}"
        )
    return ", ".join(items)


def wav_seconds(path: str) -> float:
    import wave

    with wave.open(path, "rb") as wav:
        return round(wav.getnframes() / float(wav.getframerate() or 24000), 2)


def synthesize_mp3(text: str, instruct: str | None, seed: int) -> tuple[bytes, dict]:
    t0 = time.monotonic()
    uid = uuid.uuid4().hex
    wav_path = f"/tmp/tts-{uid}.wav"
    mp3_path = f"/tmp/tts-{uid}.mp3"
    argv = [
        TTS_BIN, "--model", BASE_GGUF, "--codec", CODEC_GGUF,
        "--lang", LANG, "--seed", str(seed), "-o", wav_path,
    ]
    if instruct:
        argv += ["--instruct", instruct]
    try:
        with _GPU_LOCK:
            t1 = time.monotonic()
            proc = subprocess.run(  # noqa: S603 - argv built internally, no shell
                argv, input=text.encode("utf-8"), capture_output=True,
                timeout=TIMEOUT_S, check=False,
            )
            gen_ms = int((time.monotonic() - t1) * 1000)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace")[-400:]
            raise RuntimeError(f"tts exit={proc.returncode}: {tail}")
        audio_s = wav_seconds(wav_path)
        t2 = time.monotonic()
        enc = subprocess.run(  # noqa: S603
            ["ffmpeg", "-loglevel", "error", "-i", wav_path,
             "-codec:a", "libmp3lame", "-b:a", BITRATE, "-y", mp3_path],
            capture_output=True, timeout=60, check=False,
        )
        enc_ms = int((time.monotonic() - t2) * 1000)
        if enc.returncode != 0:
            raise RuntimeError(f"ffmpeg exit={enc.returncode}: "
                               f"{enc.stderr.decode('utf-8', 'replace')[-300:]}")
        with open(mp3_path, "rb") as f:
            audio = f.read()
        meta = {
            "X-Gen-Ms": str(gen_ms),
            "X-Encode-Ms": str(enc_ms),
            "X-Audio-Seconds": str(audio_s),
            "X-Chars": str(len(text)),
            "X-Bitrate": BITRATE,
        }
        archive_save(audio, text=text, voice=instruct, seed=seed,
                     audio_s=audio_s, gen_ms=gen_ms)
        return audio, meta
    finally:
        for p in (wav_path, mp3_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code: int, body: dict, extra: dict | None = None) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/tts":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1 << 20:
                self._json(413, {"error": "body too large"})
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            text = normalize_persian(str(body.get("text", "")))
            instruct = validate_instruct(body.get("instruct"))
            seed_raw = body.get("seed")
            seed = int(seed_raw) if isinstance(seed_raw, (int, str)) and \
                str(seed_raw).lstrip("-").isdigit() else -1
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(422, {"error": str(exc), "valid_instruct": sorted(INSTRUCT_ITEMS)})
            return
        try:
            audio, meta = synthesize_mp3(text, instruct, seed)
        except subprocess.TimeoutExpired:
            self._json(504, {"error": f"synthesis exceeded {TIMEOUT_S}s"})
            return
        except RuntimeError as exc:
            self._json(502, {"error": str(exc)[:500]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        for k, v in meta.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(audio)
        print(f"tts done chars={len(text)} gen={meta['X-Gen-Ms']}ms "
              f"enc={meta['X-Encode-Ms']}ms audio={meta['X-Audio-Seconds']}s", flush=True)

    def log_message(self, fmt, *args) -> None:  # quieter default logs
        pass


if __name__ == "__main__":
    print(f"tts-worker on :{PORT} model={BASE_GGUF} bitrate={BITRATE}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
