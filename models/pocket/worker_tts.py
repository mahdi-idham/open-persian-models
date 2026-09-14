"""pocket-worker: pocket-tts-farsi v1 CPU HTTP server (stdlib + pocket-tts).

POST /tts {text, seed?}
  -> normalize_fa -> split_text (<=18-token chunks) -> pocket-tts per chunk
     (runaway recovery) -> join with pauses -> ffmpeg -> MP3 -> bytes
GET /health -> {"ok": true}

Contract mirrors models/omnivoice in this repo, so a future router
only swaps the upstream URL. Differences, all deliberate:
- CPU persistent model (100M params, loaded ONCE at boot — no per-call reload;
  threading.Lock still serializes synthesis).
- Voice = ONE baked prompt (example_voice.wav); `instruct` accepted-but-IGNORED.
- pocket-tts has NO seed: `seed` accepted-but-IGNORED, every call is a fresh
  stochastic take (reroll = just call again). Never send seed=0-meaning-random
  confusion here — seeds do nothing at all.
- Text norm = farsi_tts.normalize (the EXACT training normalization:
- Arabic folds, digits-to-words, harakat strip).
- Fully OFFLINE: the hf:// URIs inside farsi.yaml are rewritten to the local
  /models/tts/pocket-fa/ files at boot (HF_HUB_OFFLINE=1 in the Dockerfile).

Env: MODEL_YAML VOICE_WAV POCKET_TEMP=0.3 POCKET_EOS=-2.0
     BITRATE=64k TIMEOUT_S=120 PORT=8302 MAX_CHARS=1500 ARCHIVE_DIR=/archive
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_YAML = os.environ.get("MODEL_YAML", "/models/tts/pocket-fa/farsi.yaml")
VOICE_WAV = os.environ.get("VOICE_WAV", "/models/tts/pocket-fa/example_voice.wav")
TEMP = float(os.environ.get("POCKET_TEMP", "0.3"))
EOS = float(os.environ.get("POCKET_EOS", "-2.0"))
BITRATE = os.environ.get("BITRATE", "64k")
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "120"))
PORT = int(os.environ.get("PORT", "8302"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1500"))
MAX_TOKENS = 18          # author-measured clean ceiling (farsi_tts.py)
PAUSE_SEC = 0.15         # pause after punctuation-ending chunk (farsi_tts default)
JOIN_SEC = 0.15          # seam for mid-phrase splits (farsi_tts --join-sec default)

_GPU_LOCK = threading.Lock()

ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "").strip()
_TEHRAN = timezone(timedelta(hours=3, minutes=30))

def _localize_config(yaml_path: str) -> str:
    """Rewrite hf:// weight/tokenizer URIs to files sitting next to the yaml.

    farsi.yaml pins `hf://mehdi-hf/pocket-tts-farsi/<file>@<sha>` — handing it to
    load_model makes pocket-tts DOWNLOAD those blobs (on filtered networks the
    container silently re-pulls ~438 MB and hangs). The local bundle in
    /models/tts/pocket-fa/ IS those pinned blobs (sha256-verified at build
    time), so point at them and stay offline.
    """
    d = os.path.dirname(os.path.abspath(yaml_path))
    with open(yaml_path, encoding="utf-8") as f:
        txt = f.read()

    def fix_line(m):
        indent, key, uri = m.group(1), m.group(2), m.group(3).strip()
        fname = uri.split("/")[-1].split("@")[0]
        return "%s%s: %s" % (indent, key, os.path.join(d, fname))

    out, n = re.subn(r"(?m)^([ \t]*)(weights_path|tokenizer_path):[ \t]*(hf://\S+)[ \t]*$",
                     fix_line, txt)
    missing = [p for p in re.findall(r"(?m)^[ \t]*(?:weights_path|tokenizer_path):[ \t]*(/\S+)[ \t]*$", out)
               if not os.path.exists(p)]
    # only code lines matter: the yaml header comments mention hf:// by design
    code = "\n".join(l for l in out.splitlines() if not l.lstrip().startswith("#"))
    if n != 2 or "hf://" in code:
        raise RuntimeError("config localize failed (rewrote %d, hf:// left=%s)"
                           % (n, "hf://" in code))
    if missing:
        raise RuntimeError("localized config missing files: %s" % missing)
    tmp = "/tmp/farsi-local.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(out)
    print("config localized: %d hf:// path(s) -> %s" % (n, d), flush=True)
    return tmp


print("loading pocket-tts model ...", flush=True)
from pocket_tts import TTSModel  # noqa: E402
from normalize_fa import normalize  # noqa: E402  (exact training norm)
from farsi_tts import split_text, generate_chunk, BREAK_CHARS  # noqa: E402

import numpy as np  # noqa: E402

_t0 = time.monotonic()
MODEL = TTSModel.load_model(config=_localize_config(MODEL_YAML), temp=TEMP, eos_threshold=EOS)
STATE = MODEL.get_state_for_audio_prompt(VOICE_WAV)
SR = int(MODEL.mimi.sample_rate)
_SP = MODEL.flow_lm.conditioner.tokenizer.sp
print("pocket ready in %.1fs sr=%d voice=%s" %
      (time.monotonic() - _t0, SR, os.path.basename(VOICE_WAV)), flush=True)


def archive_save(mp3: bytes, *, text: str, audio_s: float,
                 gen_ms: int) -> str | None:
    """Save MP3 + JSON sidecar to ARCHIVE_DIR. Never raises."""
    if not ARCHIVE_DIR:
        return None
    try:
        os.makedirs(ARCHIVE_DIR, mode=0o755, exist_ok=True)
        now = datetime.now(_TEHRAN)
        stamp = now.strftime("%Y%m%d-%H%M%S")
        base = "%s_pocket_%dch_%s" % (stamp, len(text), uuid.uuid4().hex[:6])
        mp3_path = os.path.join(ARCHIVE_DIR, base + ".mp3")
        json_path = os.path.join(ARCHIVE_DIR, base + ".json")
        with open(mp3_path, "wb") as f:
            f.write(mp3)
        sidecar = {
            "ts": time.time(),
            "day_tehran": now.strftime("%Y-%m-%d"),
            "model": "pocket-fa-v1",
            "voice": os.path.basename(VOICE_WAV),
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


def count_tokens(s: str) -> int:
    return len(_SP.encode(s))


def synthesize_mp3(text: str) -> tuple[bytes, dict]:
    cleaned = normalize(text)
    if not cleaned or not cleaned.strip():
        raise ValueError("text is empty after normalization")
    if len(cleaned) > MAX_CHARS:
        raise ValueError("text is %d chars, limit is %d" % (len(cleaned), MAX_CHARS))
    chunks = split_text(cleaned, count_tokens, MAX_TOKENS)
    uid = uuid.uuid4().hex
    wav_path = "/tmp/tts-%s.wav" % uid
    mp3_path = "/tmp/tts-%s.mp3" % uid
    try:
        with _GPU_LOCK:
            t1 = time.monotonic()
            pieces = []
            for i, chunk in enumerate(chunks, 1):
                pieces.append(generate_chunk(MODEL, STATE, chunk, SR))
                if i < len(chunks):
                    seam = PAUSE_SEC if chunk.rstrip().endswith(tuple(BREAK_CHARS)) \
                        else JOIN_SEC
                    pieces.append(np.zeros(int(seam * SR), dtype=np.float32))
            wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            gen_ms = int((time.monotonic() - t1) * 1000)
        import soundfile as sf
        sf.write(wav_path, wav, SR)
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
            audio = f.read()
        meta = {
            "X-Gen-Ms": str(gen_ms),
            "X-Encode-Ms": str(enc_ms),
            "X-Audio-Seconds": str(audio_s),
            "X-Chars": str(len(cleaned)),
            "X-Bitrate": BITRATE,
            "X-Model": "pocket-fa-v1",
            "X-Chunks": str(len(chunks)),
        }
        archive_save(audio, text=cleaned, audio_s=audio_s, gen_ms=gen_ms)
        return audio, meta
    finally:
        for p in (wav_path, mp3_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
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
            text = str(body.get("text", ""))
            if not text.strip():
                raise ValueError("text is required (non-empty string)")
            # seed/instruct accepted-but-IGNORED (no such concepts here).
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(422, {"error": str(exc)})
            return
        try:
            audio, meta = synthesize_mp3(text)
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
        print("tts done chars=%s chunks=%s gen=%sms audio=%ss" %
              (meta["X-Chars"], meta["X-Chunks"], meta["X-Gen-Ms"],
               meta["X-Audio-Seconds"]), flush=True)

    def log_message(self, fmt, *args) -> None:
        pass


if __name__ == "__main__":
    print("pocket-worker on :%d voice=%s" % (PORT, VOICE_WAV), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
