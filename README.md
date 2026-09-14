# داکر فارسی TTS

مجموعه داکرفایل‌های آماده برای ساخت سرویس تبدیل متن به گفتار (TTS) روی کارت گرافیک.
هر مدل، پوشه خودش را دارد. نسخه زنده: [فارسی‌هوش](https://farsihoosh.ir)

## این ریپو چیست؟

اگر یک مدل TTS دارید و می‌خواهید آن را داخل داکر اجرا کنید، لازم نیست از صفر شروع کنید.
اینجا برای هر مدل یک پوشه هست با سه چیز: `Dockerfile` + ورکر پایتون + راهنمای همان مدل.
پوشه `models/_template` هم قالب خالی برای مدل بعدی است.

## مدل‌های موجود

| مدل | چه کار می‌کند | پوشه |
|---|---|---|
| omnivoice | تبدیل متن فارسی به گفتار با چند صدا (زن، مرد، کودک و...) | [models/omnivoice](models/omnivoice) |
| pocket | تبدیل متن فارسی به گفتار با یک صدا، فقط CPU (بدون گرافیک)، سریع | [models/pocket](models/pocket) |

مدل جدید اضافه شد، همین‌جا در جدول می‌آید. راهنمای افزودن مدل در [قالب](models/_template) است.

## پیش‌نیازها

- لینوکس (برای مدل omnivoice: کارت گرافیک NVIDIA حداقل ۸ گیگ VRAM؛ مدل pocket فقط CPU می‌خواهد)
- درایور NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
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

## چطور کار می‌کند؟

```
POST /tts {text} → ورکر → انجین TTS ( host GPU ) → WAV → ffmpeg → MP3 64k → جواب
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

MIT — آزاد استفاده و تغییر بدهید.
