# مدل omnivoice (نمونه فارسی)

انجین: [omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp) در ریویژن پین‌شده
`040c8b3` (بیلد reproducible — هیچ‌وقت برنچ متحرک را دنبال نکنید).

## صداها

پارامتر `instruct` فقط این موارد را قبول می‌کند (بقیه خطای 422):

`female` · `male` · `child` · `teenager` · `young adult` · `middle-aged` · `elderly` ·
`high pitch` · `moderate pitch` · `low pitch` · `very high pitch` · `very low pitch` · `whisper`

مثال: `{"text":"...","instruct":"female, young adult","seed":777}` — سید ثابت
نتیجه ثابت می‌دهد؛ سید نفرستید یعنی تصادفی.

## محدودیت‌ها

- متن حداکثر ۱۵۰۰ کاراکتر، حداقل معنادار (متن خیلی کوتاه صدای بی‌کیفیت می‌دهد).
- هم‌زمانی ۱ (قفل GPU)؛ درخواست‌های هم‌زمان صف می‌شوند.
- خروجی همیشه MP3 مونو ۶۴کیلوبیت + هدرهای `X-Gen-Ms` و `X-Audio-Seconds`.
