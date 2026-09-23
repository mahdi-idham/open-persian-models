# مدل pocket-v2 (فارسی، دو صدا، CPU)

انجین: [پاکت‌تی‌تی‌اس (فورک)](https://github.com/mallahyari/pocket-tts) + وزن‌های
[pocket-tts-farsi v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2)
(**CC-BY-NC-4.0 — فقط غیرتجاری**؛ داده آموزشی ۹۷۳ ساعت از ۲۹۷۸ گوینده).

> مجوز: وزن‌های این مدل غیرتجاری است — استفاده شخصی/تحقیقاتی با ذکر منبع
> ([صفحه مدل](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2)). برای
> استفاده تجاری از مدل `pocket` (MIT) استفاده کنید.

## چرا pocket-v2؟

- فقط CPU می‌خواهد (بدون کارت گرافیک)، حدود ۱ گیگ رم، ~۰٫۷ ثانیه برای هر جمله.
- **دو صدا**: زن و مرد (دو پرامپت صوتی)؛ با `"instruct":"female"` یا `"male"` انتخاب می‌شود (پیش‌فرض زن).
- از نسخه یک دقیق‌تر است (WER حدود ۰٫۵۸ در برابر ۲٫۰۵) و گیرکردن وسط تولید (runaway) تقریباً ندارد.

## محدودیت‌ها

- متن حداکثر ۱۵۰۰ کاراکتر؛ متن‌های بلند به تکه‌های ~۱۸ توکنی تقسیم می‌شوند.
- این مدل با واج (phoneme) کار می‌کند — ورکر خودش متن فارسی را با مدل
  [G2P](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) به واج تبدیل می‌کند؛ شما متن فارسی عادی بفرستید.
- هم‌زمانی با قفل؛ خروجی همیشه MP3 مونو ۶۴کیلوبیت + هدرهای `X-Model: pocket-fa-v2` و `X-Voice`.

## وزن‌های مدل

وزن‌ها داخل ایمیج نیستند؛ دانلود کنید و با همین ساختار در `/models` بگذارید:

- [pocket-tts-farsi v2 on HuggingFace](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2) →
  `/models/tts/pocket-fa-v2/` (`model.safetensors` + `tokenizer_ph.model` + `model.yaml` + `normalize_fa.py` + `samples/`)
- [Homo-GE2PE-Persian-HF on HuggingFace](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) →
  `/models/g2p/Homo-GE2PE-Persian-HF/`
