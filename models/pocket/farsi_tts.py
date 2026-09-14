#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pocket-tts",
#     "soundfile",
#     "sphn",
# ]
# ///
"""Generate Farsi speech with the fine-tuned Persian Pocket TTS model.

Self-contained: run it with `uv run farsi_tts.py ...` and uv builds an
isolated environment with everything needed (torch, sentencepiece, etc.) on
first use. No cloning of the training repo, no manual pip install.

The model itself, the tokenizer and every bundled voice are pulled straight
from Hugging Face the first time they're used and cached by huggingface_hub
after that (~/.cache/huggingface) -- nothing is bundled with this script.

Examples
--------
    # default voice, default text
    uv run farsi_tts.py

    # your own text, a bundled voice by name
    uv run farsi_tts.py --text "امروز هوای تهران آفتابی است." --voice narrator_male

    # your own audio file or URL as the voice to clone
    uv run farsi_tts.py --text "..." --voice /path/to/some_persian_speaker.wav
    uv run farsi_tts.py --text "..." --voice hf://kyutai/tts-voices/alba-mackenna/casual.wav

    # a whole article
    uv run farsi_tts.py --text-file article.txt --out article.wav

    # see what's available
    uv run farsi_tts.py --list-voices
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

logger = logging.getLogger("farsi_tts")

# ---------------------------------------------------------------------------
# Configuration -- edit these two things and nothing else needs to change.
# ---------------------------------------------------------------------------

# Your Hugging Face repo. farsi.yaml there points at the actual model weights
# and tokenizer, both pinned to a commit, so this one line is the only address
# this script needs.
MODEL_CONFIG = "hf://mehdi-hf/pocket-tts-farsi/farsi.yaml"

# Named voices you can pass to --voice. Add more by uploading an audio file to
# your HF repo and adding a line here -- no other code changes needed. Every
# value is anything get_state_for_audio_prompt accepts: an hf:// URI, a plain
# https:// URL, or a local path.
VOICES: dict[str, str] = {
    "default": "hf://mehdi-hf/pocket-tts-farsi/example_voice.wav",
    # "narrator_male": "hf://mehdi-hf/pocket-tts-farsi/male_narrator.wav",
}
DEFAULT_VOICE = "default"

DEFAULT_TEXT = "کریس رایت، وزیر انرژی آمریکا، گفت واشینگتن اقتصاد جمهوری اسلامی را تحت فشار قرار می‌دهد تا سیاست حکومت تغییر کند یا حکومت جدیدی در ایران تشکیل شود."
# DEFAULT_TEXT = "سلام، حال شما چطور است؟ این صدای مصنوعی فارسی است."

# ---------------------------------------------------------------------------
# Generation defaults, chosen by ear on this model. See the model card on HF
# for why: guidance is baked in (no --cfg needed), --frames-after-eos trims a
# trailing breath the model picked up from its training data, and voice
# prompts longer than this were never seen during training.
# ---------------------------------------------------------------------------
TEMPERATURE = 0.3
EOS_THRESHOLD = -2.0
FRAMES_AFTER_EOS = 0
VOICE_PROMPT_MAX_SEC = 5.0
MAX_TOKENS_PER_CHUNK = 18  # well under pocket-tts's own 50-token limit.
# Measured on held-out speakers: chunks of 21+ tokens ran past EOS
# deterministically (the model never stops and fills to its length cap with
# repetition), while 9-16 token chunks were clean. Training utterances averaged
# ~3.8 s, about 11 tokens, so 18 stays near that distribution.
CAP_RATIO = 0.97  # healthy generations measured <=0.92 of the cap, runaways >=1.03
MAX_SPLIT_DEPTH = 2


def _cap_seconds(model, text: str) -> float:
    """The length cap pocket-tts will apply to this text, in seconds.

    Mirrors TTSModel._estimate_max_gen_len; values read from the model where
    possible, with the pocket-tts 3.0 defaults as fallback.
    """
    tokens = model.flow_lm.conditioner.prepare(text).shape[1]
    tps = getattr(model, "_TOKENS_PER_SECOND_ESTIMATE", 3.0)
    pad = getattr(model, "_GEN_SECONDS_PADDING", 2.0)
    frame_rate = model.config.mimi.frame_rate
    return math.ceil((tokens / tps + pad) * frame_rate) / frame_rate


def generate_chunk(model, state, text, sample_rate, retries=1, seam_sec=0.05, _depth=0):
    """Generate one chunk, recovering from runaway (no-EOS) generations.

    A generation that never emits EOS runs to the length cap and fills the tail
    with repetition. Two mitigations, cheapest first:

    1. Retry -- sampling is stochastic, so marginal chunks often terminate on a
       second attempt (2 of 5 runaways recovered this way when measured).
    2. Split and recurse -- the rest are stuck deterministically: every seed
       produces exactly cap-length audio. Halving the text fixed all of them,
       since both halves land back inside the trained distribution.

    Lowering eos_threshold is NOT a fix: stuck chunks ignored it down to -6.0,
    and where it did fire it truncated to a third of the expected length.

    Kept in sync with training/farsi/synthesize.py in the pocket-tts repo.
    """
    import numpy as np  # imported lazily here, matching main()'s pattern

    cap = _cap_seconds(model, text)
    shortest = None
    for _ in range(retries + 1):
        audio = np.asarray(
            model.generate_audio(state, text, frames_after_eos=FRAMES_AFTER_EOS), dtype=np.float32
        ).reshape(-1)
        if len(audio) / sample_rate <= CAP_RATIO * cap:
            return audio
        if shortest is None or len(audio) < len(shortest):
            shortest = audio

    words = text.split()
    if _depth >= MAX_SPLIT_DEPTH or len(words) < 4:
        logger.warning(f"chunk still hit the length cap after splitting: {text}")
        return shortest if shortest is not None else np.zeros(1, dtype=np.float32)

    half = len(words) // 2
    logger.info(f"chunk hit the length cap; splitting {len(words)} words -> {half}+{len(words) - half}")
    seam = np.zeros(int(seam_sec * sample_rate), dtype=np.float32)
    parts = []
    for piece in (" ".join(words[:half]), " ".join(words[half:])):
        if parts:
            parts.append(seam)
        parts.append(
            generate_chunk(model, state, piece, sample_rate, retries, seam_sec, _depth + 1)
        )
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Minimal Persian text splitting: pocket-tts's own chunker only splits at
# sentence boundaries, which does nothing for one long comma-joined sentence.
# This mirrors training/farsi/synthesize.py in the pocket-tts repo.
# ---------------------------------------------------------------------------
HARD_BREAK = re.compile(r"(?<=[.!؟])\s+")
SOFT_BREAK = re.compile(r"(?<=[،؛:])\s+")
BREAK_CHARS = (".", "!", "؟", "،", "؛", ":")


def split_text(
    text: str,
    count_tokens,
    max_tokens: int,
    keep_punct_boundaries: bool = False,
    min_tokens: int = 8,
) -> list[str]:
    """Split `text` into chunks of at most `max_tokens`, breaking at sentence
    punctuation first, then clause punctuation, then words as a last resort.
    Neighbouring chunks are merged back together while they still fit.

    With `keep_punct_boundaries`, every sentence/clause mark becomes a chunk
    boundary even when the text would have fitted in one chunk -- only
    boundaries get a real pause (see --pause-sec/--join-sec), so this is what
    makes a comma audible when the model rushes it. Tested and NOT the
    default: it leaves a chunk ending mid-clause at a comma, which no
    training utterance ever did, and that chunk can degrade reproducibly.
    """
    text = " ".join(text.split())
    if not text:
        return []

    def fits(s: str) -> bool:
        return count_tokens(s) <= max_tokens

    def split_by(pattern, piece: str) -> list[str]:
        parts = [p.strip() for p in pattern.split(piece) if p.strip()]
        return parts if len(parts) > 1 else []

    def recurse(piece: str) -> list[str]:
        if fits(piece):
            return [piece]
        for pattern in (HARD_BREAK, SOFT_BREAK):
            parts = split_by(pattern, piece)
            if parts:
                return [c for p in parts for c in recurse(p)]
        out, cur = [], ""
        for word in piece.split():
            trial = f"{cur} {word}".strip()
            if cur and not fits(trial):
                out.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            out.append(cur)
        return out

    if keep_punct_boundaries:
        pieces = [text]
        for pattern in (HARD_BREAK, SOFT_BREAK):
            pieces = [q for piece in pieces for q in (split_by(pattern, piece) or [piece])]
        parts = [c for piece in pieces for c in recurse(piece)]
    else:
        parts = recurse(text)

    merged: list[str] = []
    for chunk in parts:
        # A punctuation boundary is only worth keeping if the chunk before it
        # is long enough to synthesize well -- a two-word opener rendered
        # right after the voice prompt tends to come out badly.
        keep_apart = (
            keep_punct_boundaries
            and merged
            and merged[-1].rstrip().endswith(BREAK_CHARS)
            and count_tokens(merged[-1]) >= min_tokens
        )
        if merged and not keep_apart and fits(f"{merged[-1]} {chunk}"):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


# ---------------------------------------------------------------------------
# Persian text normalization: normalize_fa.py, copied verbatim (not
# retyped -- hand-porting a regex full of near-identical Arabic combining
# marks is exactly how a real bug got introduced here once) from the training
# repo's training/farsi/normalize_fa.py. Keep normalize_fa.py next to this
# script; to update, just re-copy that file over this one.
# ---------------------------------------------------------------------------
from normalize_fa import normalize  # noqa: E402


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_voice(voice: str) -> str:
    if voice in VOICES:
        return VOICES[voice]
    return voice  # a path, an hf:// URI, or a plain URL -- pocket-tts resolves it


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=None, help="text to speak")
    parser.add_argument("--text-file", default=None, help="read the text from this file instead")
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"a name from --list-voices, a local audio file, or an hf://.../https:// URL (default: {DEFAULT_VOICE})",
    )
    parser.add_argument("--out", default=None, help="output .wav (default: <script dir>/output.wav)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--eos-threshold", type=float, default=EOS_THRESHOLD)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_CHUNK)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=8,
        help="never leave a chunk shorter than this many tokens (a short fragment "
        "synthesizes badly since the model trained on ~4 second utterances)",
    )
    parser.add_argument(
        "--voice-sec",
        type=float,
        default=VOICE_PROMPT_MAX_SEC,
        help=f"seconds of the voice prompt to use; 0 uses the whole file (default: {VOICE_PROMPT_MAX_SEC})",
    )
    parser.add_argument(
        "--pause-sec", type=float, default=0.15, help="silence after a chunk that ends at punctuation"
    )
    parser.add_argument(
        "--join-sec",
        type=float,
        default=0.15,
        help="silence at a mid-phrase split with no punctuation (chunks are generated "
        "independently, so 0 here exposes an audible seam)",
    )
    parser.add_argument(
        "--pause-at-punct",
        action="store_true",
        help="make every . ! ؟ ، ؛ : a chunk boundary so each gets --pause-sec. NOT "
        "recommended on this model: tested and found to reproducibly degrade a chunk "
        "left ending mid-clause at a comma, which no training utterance ever did.",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="skip Persian text normalization (not recommended)"
    )
    parser.add_argument("--list-voices", action="store_true", help="print the named voices and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.list_voices:
        print("Named voices (pass any of these to --voice):")
        for name, src in VOICES.items():
            marker = "  (default)" if name == DEFAULT_VOICE else ""
            print(f"  {name}{marker}\n    {src}")
        print("\nYou can also pass a local file path or any hf:// / https:// URL directly.")
        return

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = DEFAULT_TEXT
        logger.info(f"no --text given, using the default sentence")
    if not args.no_normalize:
        text = normalize(text)

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / "output.wav"

    logger.info("loading model (first run downloads it from Hugging Face and caches it)...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(
        config=MODEL_CONFIG,
        temp=args.temperature,
        eos_threshold=args.eos_threshold,
    )

    voice_src = resolve_voice(args.voice)
    logger.info(f"voice: {args.voice}" + (f" -> {voice_src}" if voice_src != args.voice else ""))

    # Trim the prompt to what the model actually saw during training: longer
    # prompts are out of distribution and measurably less stable.
    import sphn

    voice_path = voice_src
    if args.voice_sec > 0:
        try:
            wav, sr = sphn.read(voice_src)
            keep = int(args.voice_sec * sr)
            if wav.shape[-1] > keep:
                trimmed = Path(tempfile.mkdtemp()) / "voice_prompt.wav"
                sphn.write_wav(str(trimmed), wav.mean(axis=0)[:keep].astype("float32"), int(sr))
                logger.info(f"voice prompt trimmed {wav.shape[-1] / sr:.1f}s -> {args.voice_sec:.1f}s")
                voice_path = str(trimmed)
        except Exception as exc:  # noqa: BLE001 -- a remote/safetensors voice: let pocket-tts handle it as-is
            logger.debug(f"could not pre-trim voice prompt ({exc}); using it unmodified")

    state = model.get_state_for_audio_prompt(voice_path)

    sp = model.flow_lm.conditioner.tokenizer.sp
    chunks = split_text(
        text, lambda s: len(sp.encode(s)), args.max_tokens, args.pause_at_punct, args.min_tokens
    )
    logger.info(f"{len(chunks)} chunk(s) to generate")

    import numpy as np

    sample_rate = int(model.mimi.sample_rate)
    pause = np.zeros(int(args.pause_sec * sample_rate), dtype=np.float32)
    join = np.zeros(int(args.join_sec * sample_rate), dtype=np.float32)
    pieces = []
    for i, chunk in enumerate(chunks, 1):
        # Full chunk text -- truncating this log line (as an earlier version
        # did) made it look like words were being dropped from generation
        # when they were only being dropped from the printed preview.
        logger.info(f"[{i}/{len(chunks)}] {len(sp.encode(chunk))} tokens: {chunk}")
        pieces.append(generate_chunk(model, state, chunk, sample_rate))
        if i < len(chunks):
            # A chunk ending at real punctuation gets a real pause; one split
            # mid-phrase to fit the token budget gets --join-sec instead.
            pieces.append(pause if chunk.rstrip().endswith(BREAK_CHARS) else join)

    wav_out = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    sphn.write_wav(str(out_path), wav_out, sample_rate)
    logger.info(f"wrote {out_path}  ({len(wav_out) / sample_rate:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
