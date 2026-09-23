"""Persian OCR HTTP worker (Bina-0.2-Rizeh). Single image per request."""
import io, os, sys, time
sys.path.insert(0, '/models/ocr/bina-0.2-rizeh')
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
app = FastAPI()
_ocr = None

ALLOWED_CT = ('image/jpeg', 'image/png', 'image/webp', 'image/gif')
SUFFIX = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/gif': '.gif'}


def sniff_image_type(data: bytes):
    """Resolve an image type from the magic bytes (no decode).

    Many HTTP clients do NOT send a per-part `Content-Type` inside a
    multipart body. `python-requests` omits it for a plain file object (and for a
    2-tuple), so a perfectly valid PNG was answered **415** — while `curl -F
    file=@x` (guesses from the extension) and the browser upload box (sends the
    File's own type) both worked. Sniffing makes the worker client-agnostic
    instead of failing on a header technicality."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def get_ocr():
    global _ocr
    if _ocr is None:
        from bina_page_ocr import BinaPageOCR
        _ocr = BinaPageOCR(device='gpu:0')
    return _ocr
@app.get('/health')
def health():
    return {'ok': True}
@app.post('/ocr')
async def ocr(file: UploadFile = File(...)):
    ctype = (file.content_type or '').lower()
    # An EXPLICIT, definitely-not-an-image type is rejected WITHOUT reading the
    # body. A missing type — or `application/octet-stream`, which is what an
    # extension-less file gives — falls through to the magic-byte sniff below.
    if ctype and ctype not in ALLOWED_CT and ctype != 'application/octet-stream':
        raise HTTPException(415, 'image only (jpeg/png/webp/gif)')
    data = await file.read()
    if len(data) > 3 * 1024 * 1024:
        raise HTTPException(413, 'body_too_large')
    if len(data) == 0:
        raise HTTPException(422, 'empty_file')
    if ctype not in ALLOWED_CT:
        sniffed = sniff_image_type(data)
        if sniffed is None:
            raise HTTPException(415, 'image only (jpeg/png/webp/gif)')
        ctype = sniffed
    # NOTE: byte caps don't stop decompression bombs — a 500 KB
    # JPEG can expand to hundreds of megapixels. Reject by header dims BEFORE
    # any decode (.size needs no .load(); never decode an unchecked image).
    try:
        from PIL import Image as _PILImage
        import io as _io
        with _PILImage.open(_io.BytesIO(data)) as _im:
            if _im.width * _im.height > 25_000_000:
                raise HTTPException(413, 'too_many_pixels')
    except HTTPException:
        raise
    except Exception:
        pass  # fail-open on unreadable headers; predict() will 422/500 it
    t0 = time.time()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=SUFFIX[ctype], delete=False) as tmp:
        tmp.write(data); path = tmp.name
    try:
        ocr = get_ocr()
        res = ocr.predict([path])
        out = next(iter(res), {}) if not isinstance(res, dict) else res
        if not isinstance(out, dict):
            out = {}
    finally:
        os.unlink(path)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse({'text': out.get('text', ''), 'lines': out.get('lines', []), 'gen_ms': ms})
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8340)
