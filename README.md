# هوش مصنوعی گفتار و نوشتار فارسی با داکر

مجموعه داکرفایل‌های آماده برای اجرای مدل‌های فارسی: تبدیل **متن به گفتار**
(TTS) روی کارت گرافیک یا CPU. نسخه زنده: [فارسی‌هوش](https://farsihoosh.ir)

## این ریپو چیست؟

اگر یک مدل TTS دارید و می‌خواهید آن را داخل داکر اجرا کنید، لازم نیست از صفر شروع کنید.
اینجا برای هر مدل یک پوشه هست با سه چیز: `Dockerfile` + ورکر پایتون + راهنمای همان مدل.
پوشه `models/_template` هم قالب خالی برای مدل بعدی است.

## مدل‌های موجود (TTS)

| مدل | چه کار می‌کند | مجوز وزن‌ها | پوشه |
|---|---|---|---|
| omnivoice | تبدیل متن فارسی به گفتار با چند صدا (زن، مرد، کودک و...) | غیرتجاری (CC-BY-NC-4.0) | [models/omnivoice](models/omnivoice) |
| pocket | تبدیل متن فارسی به گفتار با یک صدا، فقط CPU (بدون گرافیک)، سریع | آزاد (MIT) | [models/pocket](models/pocket) |
| pocket-v2 | تبدیل متن فارسی به گفتار با دو صدا (زن و مرد)، فقط CPU | غیرتجاری (CC-BY-NC-4.0) | [models/pocket-v2](models/pocket-v2) |

مدل جدید اضافه شد، همین‌جا در جدول می‌آید. راهنمای افزودن مدل در [قالب](models/_template) است.

## تشخیص متن تصویر (OCR)

فارسی‌هوش علاوه بر گفتار، استخراج متن از عکس هم دارد (مدل
[Bina-0.2-Rizeh](https://huggingface.co/Reza2kn/Bina-0.2-Rizeh)، مجوز آزاد
Apache-2.0): صفحه امتحان [persian-ocr](https://farsihoosh.ir/persian-ocr) و
مستندات [api-docs-ocr](https://farsihoosh.ir/api-docs-ocr).
داکر آماده OCR هنوز در این ریپو نیست — فعلاً فقط وزن‌ها در لینک بالا.

## پیش‌نیازها

- لینوکس (برای مدل omnivoice: کارت گرافیک NVIDIA حداقل ۸ گیگ VRAM؛ مدل‌های pocket و pocket-v2 فقط CPU می‌خواهند)
- درایور NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (فقط برای omnivoice)
- داکر + Docker Compose
- فایل وزن‌های مدل (داخل ایمیج نمی‌آید؛ با volume وصل می‌شود)

## شروع سریع (مدل omnivoice)

```bash
# ۱. مدل‌ها را جایی روی هاست بگذارید، مثلا:
/models/tts/omnivoice-q4/

# ۲. ffmpeg استاتیک را کنار داکرفایل بگذارید (۷۷ مگ، یک بار):
cd models/omnivoice && bash get_ffmpeg.sh && cd ../..

# ۳. بیلد و اجرا:
docker compose up -d --build

# ۴. تست سلامت و تولید صدا:
curl localhost:8300/health
curl -s localhost:8300/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا","instruct":"female, young adult"}' -o out.mp3
```

## شروع سریع (مدل pocket، فقط CPU)

```bash
# ۱. وزن‌ها را جایی روی هاست بگذارید، مثلا:
/models/tts/pocket-fa/

# ۲. ffmpeg استاتیک را کنار داکرفایل بگذارید (۷۷ مگ، یک بار):
cd models/pocket && bash get_ffmpeg.sh && cd ../..

# ۳. بیلد و اجرا (بدون نیاز به GPU):
docker compose up -d --build tts-pocket

# ۴. تست سلامت و تولید صدا:
curl localhost:8302/health
curl -s localhost:8302/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا"}' -o out.mp3
```

## شروع سریع (مدل pocket-v2، فقط CPU، دو صدا)

```bash
# ۱. وزن‌ها را جایی روی هاست بگذارید، مثلا:
/models/tts/pocket-fa-v2/
# و مدل G2P (تبدیل متن به واج):
/models/g2p/Homo-GE2PE-Persian-HF/

# ۲. ffmpeg استاتیک را کنار داکرفایل بگذارید (۷۷ مگ، یک بار):
cd models/pocket-v2 && bash get_ffmpeg.sh && cd ../..

# ۳. بیلد و اجرا (بدون نیاز به GPU):
docker compose up -d --build tts-pocketv2

# ۴. تست سلامت و تولید صدا (زن و مرد):
curl localhost:8303/health
curl -s localhost:8303/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا","voice":"female"}' -o out-f.mp3
curl -s localhost:8303/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا","voice":"male"}' -o out-m.mp3
```

## چطور کار می‌کند؟

```
POST /tts {text} → ورکر → انجین TTS (GPU یا CPU) → WAV → ffmpeg → MP3 64k → جواب
GET /health → {"ok": true}
```

- ورکر با پایتون خالص نوشته شده (بدون فریم‌ورک) و یک قفل دارد: هر بار فقط یک سنتز.
- متن فارسی قبل از تولید نرمالایز می‌شود (حروف عربی به فارسی، حذف تشکیلات اضافه).
- خروجی MP3 مونو ۶۴کیلوبیت است؛ برای اینترنت کم‌سرعت بهینه شده.

## افزودن مدل جدید

۱. `cp -r models/_template models/<model-name>`
۲. داکرفایل را پر کنید (سورس انجین + SHA پین‌شده + دستور بیلد).
۳. ورکر را بنویسید (دو مسیر کافی است: `POST /tts` و `GET /health`).
۴. در `docker-compose.yml` یک سرویس با پورت متفاوت اضافه کنید.
۵. بیلد و تست مثل شروع سریع.

راهنمای کامل در `models/_template/README.md` است.

## نکته برای شبکه‌های فیلترشده

اگر `git clone` داخل بیلد به گیت‌هاب وصل نمی‌شود، بیلد را با پروکسی SOCKS اجرا کنید:

```bash
docker build --network=host --build-arg GIT_PROXY_URL=socks5h://127.0.0.1:10808 models/omnivoice/
```

## لایسنس

فایل‌های این ریپو MIT — آزاد استفاده و تغییر بدهید. دقت کنید **وزن مدل‌ها
مجوز خودشان را دارند**: pocket آزاد (MIT)، ولی omnivoice و pocket-v2
**فقط غیرتجاری** (CC-BY-NC-4.0) و مدل OCR آزاد (Apache-2.0) است.
