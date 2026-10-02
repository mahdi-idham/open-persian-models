"""img worker: Qwen-Image-2.1 text-to-image HTTP server behind ggk (loopback :9412).

Runs the GGUF build of Qwen-Image-2.1 on a single NVIDIA GPU with the `ggk`
engine (MIT, github.com/gguf-org/ggk) instead of diffusers.

IMPORTANT - LICENCE AND STATUS (read before using):
  * The WEIGHTS are NOT in this repo. They live on HuggingFace at
    gguf-org/qwen-image-2.1-gguf and carry the upstream Qwen licence
    ("other": review the model card before ANY commercial use). Nothing here
    changes those terms.
  * The reference deployment of this worker serves requests at $0 only. If you
    monetise a service built on it, re-check the weight licence first.
  * This folder is a PREVIEW: the image build has not been verified end to end
    in CI (see README.md for what is and is not pinned).

Shape: one image per request, 512x512, 20 steps, euler sampler, fixed seed per
request. The worker never persists renders - it serves the bytes and deletes
the temporary file.

Routes: GET /health -> {ok, engine, busy, vram_free_mib}
        POST /img {prompt, seed?} -> image/png (+ X-Seed) | 422 | 413 | 503 | 502

Tuning (all env-overridable): VRAM_MIN_MIB, EVICT_COOLDOWN_S, RETRY_WAIT_MS,
PROMPT_MAX, RENDER_TIMEOUT_S, GPU_RELAX_CONTAINER.
"""

import http.server
import json
import os
import random
import subprocess
import tempfile
import threading
import time

# SECURITY: bind to loopback by default. This worker has NO authentication,
# so anything that can reach the port can spend your GPU. Only widen it when a
# deliberate proxy/firewall sits in front.
#   docker -p 127.0.0.1:9412:9412 ...   -> keep LISTEN=0.0.0.0 inside the
#   container ONLY, because docker forwards to the container's eth0.
LISTEN_ADDR = _str_env("LISTEN", "127.0.0.1")
LISTEN_PORT = _int_env("PORT", 9412)
LISTEN = (LISTEN_ADDR, LISTEN_PORT)
MODELS = "/models"
DIFFUSION = os.path.join(MODELS, "qwen-image-2.1-nvfp4.gguf")
VAE = os.path.join(MODELS, "pig_qwen_image_2.1_vae_fp32-f16.gguf")
LLM = os.path.join(MODELS, "pig_clip-f16.gguf")
ADAPTER = os.path.join(MODELS, "pig_qwen3vl_8b_adapter-f16.gguf")


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _str_env(name, default):
    v = os.environ.get(name)
    return v if isinstance(v, str) and v else default


VRAM_MIN_MIB = _int_env("VRAM_MIN_MIB", 7000)  # measured 2026-09-29: 6417
# free OOMs, 7231 renders — the 512/20 render needs ~7 GB contiguous
# (4.7 GB single VAE alloc). Below this the worker frees VRAM (if
# GPU_RELAX_CONTAINER is set) or refuses fast with 503.
PROMPT_MAX = _int_env("PROMPT_MAX", 500)
RENDER_TIMEOUT = _int_env("RENDER_TIMEOUT_S", 300)
RETRY_WAIT_MS = _int_env("RETRY_WAIT_MS", 15000)
# SECURITY: when set, the worker runs `docker restart <name>` on this host to
# free VRAM. That requires the docker socket, which makes the container
# effectively root-equivalent on the host. Unset by default; only ever name a
# container you are willing to have stopped.
GPU_RELAX = _str_env("GPU_RELAX_CONTAINER", "")
EVICT_COOLDOWN_S = _int_env("EVICT_COOLDOWN_S", 30)
EVICT_STAMP = "/tmp/img_evict_cooldown"  # container-local

_lock = threading.Lock()


def vram_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return -1


def log(**kw):
    print(json.dumps(kw), flush=True)


def gpu_state():
    """(util_pct, used_mib) or (-1, -1). Read-only nvidia-smi."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, m = out.split(',')
        return int(u.strip()), int(m.strip())
    except Exception:
        return -1, -1


def quiet():
    """True iff no in-flight work is suspected. Signal = VRAM STABILITY
    across two samples 3 s apart (|delta| < 200 MiB) — deliberately NOT GPU
    utilization: on an actively-used laptop the desktop compositor spikes
    util to ~30% with no compute load (measured 2026-09-29, ),
    which would veto every eviction forever. Residual risk (steady-footprint
    mid-inference) is accepted and documented; cooldown + retry bound it."""
    _, m1 = gpu_state()
    time.sleep(3)
    _, m2 = gpu_state()
    if m1 < 0:
        return False
    return abs(m2 - m1) < 200


def self_evict(prompt_chars):
    """Guarded self-eviction for production (): quiet-check →
    restart other GPU model → wait until free >= threshold (cap 90 s). Returns the
    free MiB afterwards, or -1 if eviction was refused/impossible.
    ONLY other GPU model is ever touched. Cooldown between evictions."""
    try:
        if time.time() - os.path.getmtime(EVICT_STAMP) < EVICT_COOLDOWN_S:
            log(event="evict_skip", prompt_chars=prompt_chars,
                reason="cooldown")
            return -1
    except OSError:
        pass
    if not quiet():
        log(event="evict_skip", prompt_chars=prompt_chars,
            reason="not-quiet")
        return -1
    t0 = time.time()
    r = subprocess.run(['docker', 'restart', GPU_RELAX],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log(event="evict_fail", prompt_chars=prompt_chars,
            reason="restart-failed")
        return -1
    free = -1
    while time.time() - t0 < 90:
        time.sleep(3)
        free = vram_free_mib()
        if free >= VRAM_MIN_MIB:
            break
    ms = int((time.time() - t0) * 1000)
    try:
        open(EVICT_STAMP, 'w').write(str(int(time.time())))
    except OSError:
        pass
    log(event="evict", prompt_chars=prompt_chars, ms=ms,
        vram_free_mib=free,
        ok=free >= VRAM_MIN_MIB)
    return free if free >= VRAM_MIN_MIB else -1


class H(http.server.BaseHTTPRequestHandler):
    server_version = "img-worker/1"

    def _json(self, code, obj, extra=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self._json(404, {"error": "not_found"})
        busy = _lock.locked()
        self._json(200, {"ok": True, "engine": "ggk/cuda0",
                         "busy": busy, "vram_free_mib": vram_free_mib()})

    def do_POST(self):
        if self.path != "/img":
            return self._json(404, {"error": "not_found"})
        try:
            ln = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json(422, {"error": "bad_length"})
        if ln > 4096:
            return self._json(413, {"error": "body_too_large"})
        try:
            req = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            return self._json(422, {"error": "bad_json"})
        prompt = req.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            return self._json(422, {"error": "prompt_required"})
        if len(prompt) > PROMPT_MAX:
            return self._json(413, {"error": "prompt_too_long"})
        raw_seed = req.get("seed")
        if raw_seed is None:
            # No seed sent: roll one per request — ggk's own default is a
            # FIXED seed (42), which renders near-identical images for the
            # same prompt (owner report 2026-10-01). The relay always sends
            # a seed; this covers direct/bench callers too.
            seed = random.randint(0, 999999)
        elif (isinstance(raw_seed, bool) or not isinstance(raw_seed, int)
                or not 0 <= raw_seed <= 999999):
            return self._json(422, {"error": "bad_seed"})
        else:
            seed = raw_seed
        if _lock.locked():
            return self._json(503, {"error": "gpu_busy"},
                               {"Retry-After": "30"})
        # NOTE: no pre-flight refuse here — eviction below frees the GPU
        # first (); refuse happens only if eviction can't.
        if not _lock.acquire(blocking=False):
            return self._json(503, {"error": "gpu_busy"},
                               {"Retry-After": "30"})
        # Lock is held from here through evict + render + any retry — the
        # single-render invariant is never broken.
        try:
            free = vram_free_mib()
            if free >= 0 and free < VRAM_MIN_MIB:
                # Production behavior (): free the GPU ourselves
                # (other GPU model-only, guarded) instead of refusing — target state is
                # screen + driver + idle contexts only.
                free = self_evict(len(prompt))
                if free < 0:
                    # Eviction skipped/failed: re-read; refusing fast beats
                    # burning 25 s on a still-full GPU.  retest.
                    free = vram_free_mib()
            if free >= 0 and free < VRAM_MIN_MIB:
                log(event="refuse", prompt_chars=len(prompt),
                    vram_free_mib=free, status=503)
                return self._json(503, {"error": "gpu_busy",
                                        "vram_free_mib": free},
                                   {"Retry-After": "30"})
            data = self._generate(prompt, seed, free, attempt=1)
            if data is None and self._last_fail == "oom":
                # Wait out the transient contention first (TTS/OCR jobs last
                # seconds), THEN re-gate: refusing fast beats burning 21 s
                # on a still-full GPU. .
                log(event="retry_wait", prompt_chars=len(prompt),
                    wait_ms=RETRY_WAIT_MS)
                time.sleep(RETRY_WAIT_MS / 1000)
                free2 = vram_free_mib()
                if free2 >= 0 and free2 < VRAM_MIN_MIB:
                    log(event="refuse", prompt_chars=len(prompt),
                        vram_free_mib=free2, status=503, after="retry_wait")
                    return self._json(503, {"error": "gpu_busy",
                                            "vram_free_mib": free2},
                                       {"Retry-After": "30"})
                data = self._generate(prompt, seed, free2, attempt=2)
            if data is None:
                return self._json(502, {"error": "img_failed"})
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Seed", str(seed))
            self.end_headers()
            self.wfile.write(data)
        finally:
            _lock.release()

    # Substrings marking a TRANSIENT (contention-class) ggk failure.
    OOM_MARKS = ("out of memory", "failed to allocate", "cumemalloc",
                 "cuda error", "alloc failed")

    def _generate(self, prompt, seed, free, attempt):
        """One render. Returns PNG bytes, or None (sets self._last_fail to
        'oom' | 'other'). Never logs raw subprocess output (may echo prompt)."""
        t0 = time.monotonic()
        fd, tmp = tempfile.mkstemp(suffix=".png", dir="/tmp")
        os.close(fd)  # mkstemp creates 0600 atomically; mktemp() has a race
        try:
            p = subprocess.run(
                ["ggk", "diffuser", "engine", "--",
                 "--diffusion-model", DIFFUSION, "--vae", VAE,
                 "--llm", LLM, "--llm-adapter", ADAPTER,
                 "--sampling-method", "euler", "--diffusion-fa",
                 "-p", prompt, "--seed", str(seed),
                 "-W", "512", "-H", "512",
                 "--steps", "20", "-o", tmp],
                capture_output=True, text=True, timeout=RENDER_TIMEOUT)
            ms = int((time.monotonic() - t0) * 1000)
            if p.returncode == 0 and os.path.exists(tmp):
                log(event="render", attempt=attempt,
                    prompt_chars=len(prompt), seed=seed, ms=ms,
                    vram_free_mib=free, status=200)
                with open(tmp, "rb") as fh:
                    return fh.read()
            blob = (p.stdout + p.stderr).lower()
            self._last_fail = ("oom" if any(m in blob for m in self.OOM_MARKS)
                               else "other")
            log(event="render", attempt=attempt,
                prompt_chars=len(prompt), ms=ms,
                vram_free_mib=free, status=502,
                fail_class=self._last_fail)
            return None
        except subprocess.TimeoutExpired:
            self._last_fail = "other"  # 300 s already blew the wait budget
            log(event="render", attempt=attempt,
                prompt_chars=len(prompt), ms=-1,
                vram_free_mib=free, status=502, fail_class="timeout")
            return None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    for f in (DIFFUSION, VAE, LLM, ADAPTER):
        assert os.path.exists(f), "missing model file: %s" % f
    srv = http.server.ThreadingHTTPServer(LISTEN, H)
    srv.daemon_threads = True
    print("img worker listening on %s:%d (no auth - keep it local)" % LISTEN, flush=True)
    srv.serve_forever()
