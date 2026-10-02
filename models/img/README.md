# qwen-image-2.1-gguf - Persian-capable text-to-image worker

Generates a 512x512 image from a text prompt, on a single NVIDIA GPU, using the
**GGUF** build of Qwen-Image-2.1 and the `ggk` engine (MIT,
[gguf-org/ggk](https://github.com/gguf-org/ggk)) instead of diffusers.

> **Status: preview.** The worker code is complete and sanitized, but the image
> build has **not** been verified end to end in CI. The text-encoder dependency
> used by the fast configuration is **not pinned** in the Dockerfile. Expect to
> finish that wiring yourself; please open an issue if you succeed.

## Security notes - read before deploying

This worker is a **local development component**. Please read all five points.

1. **There is no authentication.** Anyone who can reach the port can submit
   prompts and consume your GPU, and can read whatever the model returns. It
   binds to `127.0.0.1` by default for that reason. Put an authenticated
   reverse proxy in front before it is reachable by anything you do not control.
2. **The port is `9412` and it is configurable** via `PORT`, but a non-default
   port is obscurity, not access control. Do not rely on it.
3. **Rate limiting is your job.** There is none. A loop of concurrent requests
   will queue on the GPU rather than being rejected. Cap it at your edge.
4. **`GPU_RELAX_CONTAINER` is root-equivalent if you enable it.** When set, the
   worker runs `docker restart <name>` on the host, which requires mounting the
   docker socket. A container with the docker socket mounted can take over the
   host. It is **unset by default**; leave it that way unless you fully control
   the machine and understand the trade-off.
5. **The container needs the GPU passed through** (`--gpus all`), so it can read
   and write VRAM. Treat the image as trusted-only input.

## Licences - read this first

There are two separate licences here, and they are not the same thing:

| What | Licence | Where |
|---|---|---|
| This repo's code (worker, Dockerfile, docs) | MIT | see the repo `LICENSE` |
| The **model weights** | **Other / upstream Qwen terms** | [gguf-org/qwen-image-2.1-gguf](https://huggingface.co/gguf-org/qwen-image-2.1-gguf) |
| The `ggk` engine | MIT | [gguf-org/ggk](https://github.com/gguf-org/ggk) |

The weights are **not** redistributed here and their licence is **not** resolved
in this repository. Read the model card and decide for yourself. If you build a
commercial service on top of this, you are the one who has to clear it.

The reference deployment that motivated this folder served requests **free of
charge only**; that posture was chosen while the weight licence was being
reviewed.

## Run

1. Get the weights (not in this repo), then mount them read-only:

```bash
# expected layout under your host dir, mounted at /models
/models/qwen-image-2.1-gguf/...
```

2. Build and run (needs an NVIDIA GPU, roughly 8 GB VRAM):

```bash
cd models/img
docker build -t qwen-img-worker:latest .
docker run --rm --gpus all -p 127.0.0.1:9412:9412 \
  -v /your/host/dir:/models:ro qwen-img-worker:latest
```

3. Check it:

```bash
curl localhost:9412/health
```

4. Generate:

```bash
curl -s localhost:9412/img -H 'Content-Type: application/json' \
  -d '{"prompt":"a red bicycle leaning on a wall"}' -o out.png
```

## API

| Route | Body | Returns |
|---|---|---|
| `GET /health` | - | `{"ok":true,"engine":...,"busy":false,"vram_free_mib":...}` |
| `POST /img` | `{"prompt":"...","seed":123}` | `image/png` + `X-Seed` header |

Errors: `422` bad request, `413` prompt too long, `503` GPU busy / not enough
free VRAM, `502` render failed.

## Configuration

All optional, via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VRAM_MIN_MIB` | `7000` | refuse fast if free VRAM is below this |
| `EVICT_COOLDOWN_S` | `30` | wait after a busy verdict before retrying |
| `RETRY_WAIT_MS` | `15000` | pause before the one transient-failure retry |
| `PROMPT_MAX` | `500` | maximum prompt length in characters |
| `RENDER_TIMEOUT_S` | `300` | hard limit on one render |
| `LISTEN` | `127.0.0.1` | bind address; only widen behind a real proxy |
| `PORT` | `9412` | listen port |
| `GPU_RELAX_CONTAINER` | *(none)* | container to restart to free VRAM; **root-equivalent**, see Security notes |

### A note on `GPU_RELAX_CONTAINER`

On a single GPU it is common to run several models. If you set
`GPU_RELAX_CONTAINER`, this worker may **stop that container** to free VRAM
before refusing a request. It only ever stops the one container you named, but
be aware that mounting `/var/run/docker.sock` for this makes the container
effectively **root-equivalent on the host**. Use it only on a machine you
control, and prefer leaving it unset if you run one model per GPU.
