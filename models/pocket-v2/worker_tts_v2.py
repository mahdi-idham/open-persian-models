"""pocket-v2 worker: pocket-tts-farsi-v2 CPU HTTP server (CC-BY-NC-4.0, non-commercial only).

POST /tts {text} -> normalize_for_model -> sentence chunks -> G2P phonemes
                   -> pocket-tts v2 per chunk (runaway retry) -> ffmpeg MP3
GET  /health -> {"ok": true}

Same contract as the pocket v1 worker but adds a G2P phonemiser stage in front.
Contract: `seed`/`instruct` accepted-but-IGNORED (pocket has no seed).

NEVER serve this publicly: v2 weights are CC-BY-NC-4.0 (MODEL-LICENSES.md 3b).
Binds all interfaces in this image; restrict with ports (e.g. 127.0.0.1:8303) in compose.

Env: MODEL_DIR (v2 snapshot; default = HF cache snapshot_download, local-only)
     G2P_DIR (default = HF cache), VOICE_WAV (default snapshot samples/prompt_short_sentence.wav)
     PORT=8303 POCKET_TEMP=0.3 POCKET_EOS=-2.0 THREADS=2 MAX_CHARS=1500
     BITRATE=64k TIMEOUT_S=180 ARCHIVE_DIR=/archive
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8303"))
TEMP = float(os.environ.get("POCKET_TEMP", "0.3"))
EOS = float(os.environ.get("POCKET_EOS", "-2.0"))
THREADS = int(os.environ.get("THREADS", "2"))
BITRATE = os.environ.get("BITRATE", "64k")
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "180"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1500"))
MODEL_DIR_ENV = os.environ.get("MODEL_DIR", "").strip()
G2P_DIR_ENV = os.environ.get("G2P_DIR", "").strip()
V2_REPO = "mehdi-hf/pocket-tts-farsi-v2"
G2P_REPO = "mehdi-hf/Homo-GE2PE-Persian-HF"

MAX_CHUNK_CHARS = 200          # hard cap for a chunk with no punctuation
PAUSE_SEC = 0.15
_ARCHIVE = os.environ.get("ARCHIVE_DIR", "").strip()
_TEHRAN = timezone(timedelta(hours=3, minutes=30))
_LOCK = threading.Lock()

# phoneme post-map used by the v2 card (see bench_v2.py)
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})


def _resolve_dirs():
    """Return (snapshot_dir, g2p_dir). Prefer env; else HF cache, local-only."""
    if MODEL_DIR_ENV and G2P_DIR_ENV:
        return MODEL_DIR_ENV, G2P_DIR_ENV
    from huggingface_hub import snapshot_download
    snap = MODEL_DIR_ENV or snapshot_download(V2_REPO, local_files_only=True)
    g2p = G2P_DIR_ENV or snapshot_download(G2P_REPO, local_files_only=True)
    return snap, g2p


def _localize_config(snap_dir: str) -> str:
    """Rewrite hf:// weights/tokenizer URIs in model.yaml to the local files.

    Same P3-stall trap as v1: handing hf:// URIs to load_model makes pocket-tts
    re-download blobs it already has (and stall on filtered networks).
    """
    src = os.path.join(snap_dir, "model.yaml")
    with open(src, encoding="utf-8") as f:
        txt = f.read()

    def fix_line(m):
        indent, key, uri = m.group(1), m.group(2), m.group(3).strip()
        fname = uri.split("/")[-1].split("@")[0]
        return "%s%s: %s" % (indent, key, os.path.join(snap_dir, fname))

    out, n = re.subn(r"(?m)^([ \t]*)(weights_path|tokenizer_path):[ \t]*(hf://\S+)[ \t]*$",
                     fix_line, txt)
    code = "\n".join(l for l in out.splitlines() if not l.lstrip().startswith("#"))
    if n < 1 or "hf://" in code:
        raise RuntimeError("v2 config localize failed (rewrote %d)" % n)
    tmp = "/tmp/v2-local.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(out)
    return tmp


class _Runaway(logging.Handler):
    """Detects pocket-tts' 'Maximum generation length reached without EOS'."""

    def __init__(self):
        super().__init__()
        self.hit = False

    def emit(self, record):
        if "Maximum generation length" in record.getMessage():
            self.hit = True


RUNAWAY = _Runaway()
logging.getLogger("pocket_tts.models.tts_model").addHandler(RUNAWAY)

print("loading pocket-tts v2 (CPU) ...", flush=True)
_t0 = time.monotonic()
SNAP, G2P_DIR = _resolve_dirs()
sys.path.insert(0, SNAP)
from normalize_fa import normalize_for_model  # noqa: E402

from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa: E402

G2P_TOK = AutoTokenizer.from_pretrained(G2P_DIR)
G2P = T5ForConditionalGeneration.from_pretrained(G2P_DIR).eval()

from pocket_tts import TTSModel  # noqa: E402

# pocket_tts/models/tts_model.py forces torch.set_num_threads(1) at import;
# re-apply AFTER the import (measured sweet spot on i7-13650HX = 2).
import torch  # noqa: E402

torch.set_num_threads(THREADS)
print("threads=%d" % torch.get_num_threads(), flush=True)

MODEL = TTSModel.load_model(config=_localize_config(SNAP), temp=TEMP, eos_threshold=EOS)
VOICE_WAV = os.environ.get("VOICE_WAV", "").strip() or os.path.join(SNAP, "samples", "prompt_short_sentence.wav")
VOICES = {"female": VOICE_WAV}
for _pair in os.environ.get("VOICES", "").split(","):
    if "=" in _pair:
        _k, _v = _pair.split("=", 1)
        if _k.strip() and _v.strip():
            VOICES[_k.strip()] = _v.strip()
STATES = {}
_STATE_LOCK = threading.Lock()
def get_state(voice):
    # Prompt state per voice id; lazy-load + cache (thread-safe).
    st = STATES.get(voice)
    if st is None:
        with _STATE_LOCK:
            st = STATES.get(voice)
            if st is None:
                st = MODEL.get_state_for_audio_prompt(VOICES[voice])
                STATES[voice] = st
    return st
STATES["female"] = MODEL.get_state_for_audio_prompt(VOICES["female"])
SR = int(MODEL.sample_rate)
print("v2 ready in %.1fs sr=%d threads=%d" % (time.monotonic() - _t0, SR, THREADS), flush=True)


def phonemise(text: str) -> str:
    t = normalize_for_model(text).replace("؟", "").replace("?", "")
    enc = G2P_TOK([t], add_special_tokens=False, return_tensors="pt").to(G2P.device)
    with torch.no_grad():
        out = G2P.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
    raw = G2P_TOK.batch_decode(out, skip_special_tokens=True)[0].strip()
    return raw.translate(TO_PHONEMES).replace("1", "")


def split_sentences(text: str):
    """Sentence-ish chunks; hard-split any chunk longer than MAX_CHUNK_CHARS."""
    parts = re.split(r"(?<=[.؟?!])\s+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        while len(p) > MAX_CHUNK_CHARS:
            cut = p.rfind(" ", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > 0 else MAX_CHUNK_CHARS
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def _generate(ph, voice):
    """generate_audio + one runaway retry. Returns (audio, runaway_flag)."""
    state = get_state(voice)
    RUNAWAY.hit = False
    audio = MODEL.generate_audio(state, ph)
    if RUNAWAY.hit:
        RUNAWAY.hit = False
        audio2 = MODEL.generate_audio(state, ph)   # pocket has no seed: fresh take
        if not RUNAWAY.hit:
            return audio2, False
        return audio2, True
    return audio, False


def archive_save(mp3: bytes, *, text: str, audio_s: float, gen_ms: int) -> None:
    if not _ARCHIVE:
        return
    try:
        os.makedirs(_ARCHIVE, mode=0o755, exist_ok=True)
        now = datetime.now(_TEHRAN)
        base = "%s_pocketv2_%dch_%s" % (now.strftime("%Y%m%d-%H%M%S"), len(text), uuid.uuid4().hex[:6])
        with open(os.path.join(_ARCHIVE, base + ".mp3"), "wb") as f:
            f.write(mp3)
        with open(os.path.join(_ARCHIVE, base + ".json"), "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "day_tehran": now.strftime("%Y-%m-%d"),
                       "model": "pocket-fa-v2", "chars": len(text), "audio_s": audio_s,
                       "gen_ms": gen_ms, "text": text[:1500]}, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # never break synthesis on archive errors
        print("archive skip: %s: %s" % (type(exc).__name__, exc), flush=True)


def synthesize_mp3(text, voice="female"):
    if voice not in VOICES:
        raise ValueError("unknown voice %r (known: %s)" % (voice, ",".join(sorted(VOICES))))
    cleaned = normalize_for_model(text)
    if not cleaned or not cleaned.strip():
        raise ValueError("text is empty after normalization")
    if len(cleaned) > MAX_CHARS:
        raise ValueError("text is %d chars, limit is %d" % (len(cleaned), MAX_CHARS))
    chunks = split_sentences(cleaned)
    if not chunks:
        raise ValueError("text is empty after chunking")
    uid = uuid.uuid4().hex
    wav_path, mp3_path = "/tmp/v2-%s.wav" % uid, "/tmp/v2-%s.mp3" % uid
    try:
        with _LOCK:
            t1 = time.monotonic()
            pieces, runaways = [], 0
            import numpy as np
            for i, chunk in enumerate(chunks, 1):
                audio, hit = _generate(phonemise(chunk), voice)
                runaways += int(hit)
                pieces.append(audio.cpu().numpy().astype(np.float32))
                if i < len(chunks):
                    pieces.append(np.zeros(int(PAUSE_SEC * SR), dtype=np.float32))
            wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            gen_ms = int((time.monotonic() - t1) * 1000)
        try:
            import soundfile as sf
            sf.write(wav_path, wav, SR)
        except ImportError:  # bench image ships scipy, not soundfile
            import scipy.io.wavfile as _wavfile
            _wavfile.write(wav_path, SR, wav)
        audio_s = round(len(wav) / float(SR), 2)
        t2 = time.monotonic()
        enc = subprocess.run(  # noqa: S603
            ["ffmpeg", "-loglevel", "error", "-i", wav_path,
             "-ac", "1", "-codec:a", "libmp3lame", "-b:a", BITRATE, "-y", mp3_path],
            capture_output=True, timeout=60, check=False)
        enc_ms = int((time.monotonic() - t2) * 1000)
        if enc.returncode != 0:
            raise RuntimeError("ffmpeg exit=%d: %s" %
                               (enc.returncode, enc.stderr.decode("utf-8", "replace")[-300:]))
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()
        rtf = round(gen_ms / 1000.0 / audio_s, 3) if audio_s else None
        meta = {"X-Model": "pocket-fa-v2", "X-Voice": voice, "X-Gen-Ms": str(gen_ms), "X-Encode-Ms": str(enc_ms),
                "X-Audio-Seconds": str(audio_s), "X-Chars": str(len(cleaned)),
                "X-Chunks": str(len(chunks)), "X-Rtf": str(rtf),
                "X-Runaway": str(runaways), "X-Bitrate": BITRATE}
        archive_save(audio_bytes, text=cleaned, audio_s=audio_s, gen_ms=gen_ms)
        return audio_bytes, meta
    finally:
        for p in (wav_path, mp3_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self._json(200, {"ok": True}) if self.path == "/health" else self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/tts":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1 << 20:
                self._json(413, {"error": "body too large"})
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            text = str(body.get("text", ""))
            voice = str(body.get("voice", "female"))
            if not text.strip():
                raise ValueError("text is required (non-empty string)")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(422, {"error": str(exc)})
            return
        try:
            audio, meta = synthesize_mp3(text, voice)
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "synthesis exceeded %ds" % TIMEOUT_S})
            return
        except RuntimeError as exc:
            self._json(502, {"error": str(exc)[:500]})
            return
        except ValueError as exc:
            self._json(422, {"error": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        for k, v in meta.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(audio)
        print("tts ok voice=%s chars=%s chunks=%s gen=%sms audio=%ss rtf=%s runaway=%s" %
              (meta["X-Voice"], meta["X-Chars"], meta["X-Chunks"], meta["X-Gen-Ms"],
               meta["X-Audio-Seconds"], meta["X-Rtf"], meta["X-Runaway"]), flush=True)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print("pocket-v2 worker on :%d voice=%s" % (PORT, VOICE_WAV), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
