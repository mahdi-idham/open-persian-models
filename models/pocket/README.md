# مدل pocket (فارسی، سبک، CPU)

انجین: [pocket-tts](https://github.com/kyutai-labs/pocket-tts) + وزن‌های
[pocket-tts-farsi v1](https://huggingface.co/mehdi-hf/pocket-tts-farsi)
(MIT؛ داده آموزشی CC0 — مناسب استفاده تجاری).

## چرا pocket؟

- فقط CPU می‌خواهد (بدون کارت گرافیک)، حدود ۱ گیگ رم، سریع‌تر از مدل GPU نمونه.
- تک‌صدا است (صدای `example_voice.wav`)؛ `instruct` و `seed` قبول می‌کند ولی نادیده می‌گیرد (هر بار یک اجرای تازه).

## محدودیت‌ها

- متن حداکثر ۱۵۰۰ کاراکتر؛ متن‌های بلند به تکه‌های ~۱۸ توکنی تقسیم می‌شوند.
- حروف لاتین داخل متن فارسی حذف می‌شوند (نرمالایزر آموزشی) — برای متن خالص فارسی ایده‌آل است.
- هم‌زمانی ۱ (قفل)؛ خروجی همیشه MP3 مونو ۶۴کیلوبیت + هدر `X-Model: pocket-fa-v1`.

## وزن‌های مدل

وزن‌ها داخل ایمیج نیستند؛ از اینجا دانلود کنید و در `/models/tts/pocket-fa/` بگذارید:

- [pocket-tts-farsi v1 on HuggingFace](https://huggingface.co/mehdi-hf/pocket-tts-farsi)

فایل‌های لازم: `model.safetensors` + `tokenizer.model` + `farsi.yaml` + `normalize_fa.py` + `example_voice.wav`.
