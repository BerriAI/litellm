from __future__ import annotations
import base64
import hashlib
import io
from pathlib import Path
from typing import Optional


def _import_pil():
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception as e:
        raise ImportError("Pillow is required for image helpers. pip install Pillow") from e


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def compress_image(path: str, *, max_kb: int = 256, cache_dir: Optional[Path] = None) -> str:
    Image = _import_pil()
    img = Image.open(path).convert("RGB")
    key = hashlib.sha256((str(path) + str(max_kb)).encode()).hexdigest()[:16]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        outp = cache_dir / f"{key}.jpg"
        if outp.exists():
            return _data_url("image/jpeg", outp.read_bytes())
    quality = 95
    buf = io.BytesIO()
    while quality >= 30:
        buf.seek(0)
        buf.truncate(0)
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_kb * 1024:
            break
        quality -= 10
    b = buf.getvalue()
    if cache_dir:
        (cache_dir / f"{key}.jpg").write_bytes(b)
    return _data_url("image/jpeg", b)


async def fetch_remote_image(url: str, *, cache_dir: Optional[Path] = None, timeout: float = 10.0, max_kb: int = 256) -> str:
    try:
        import httpx  # type: ignore
    except Exception as e:
        raise ImportError("httpx is required for fetch_remote_image") from e
    key = hashlib.sha256((url + str(max_kb)).encode()).hexdigest()[:16]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        outp = cache_dir / f"{key}.jpg"
        if outp.exists():
            return _data_url("image/jpeg", outp.read_bytes())
    async with httpx.AsyncClient(timeout=timeout) as client:  # type: ignore
        r = await client.get(url)
        r.raise_for_status()
        content = r.content
    if len(content) <= max_kb * 1024:
        b = content
    else:
        Image = _import_pil()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        buf = io.BytesIO()
        quality = 95
        while quality >= 30:
            buf.seek(0)
            buf.truncate(0)
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= max_kb * 1024:
                break
            quality -= 10
        b = buf.getvalue()
    if cache_dir:
        (cache_dir / f"{key}.jpg").write_bytes(b)
    return _data_url("image/jpeg", b)
