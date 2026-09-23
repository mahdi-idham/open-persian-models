# Model ocr-bina (Persian OCR, GPU)

Engine: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) + weights
[Bina-0.2-Rizeh](https://huggingface.co/Reza2kn/Bina-0.2-Rizeh)
(**Apache-2.0** — permissive, commercial use allowed).

## Why ocr-bina?

- Reads Persian text from photos (receipts, pages, screenshots) on GPU.
- One image per request (JPEG/PNG/WebP/GIF, max 3 MB, max 25 megapixels).
- Returns JSON `{text, lines, gen_ms}`; explicit non-images get 415.

## Limits

- No auth in this worker itself — put it behind your own API (keys, quotas).
- Large images are capped before decode (25 MP decompression-bomb guard).

## Model weights and code

Weights and runtime code are NOT in the image; download the model repo and
mount it read-only:

- [Bina-0.2-Rizeh on HuggingFace](https://huggingface.co/Reza2kn/Bina-0.2-Rizeh) →
  `/models/ocr/bina-0.2-rizeh/`
