import io
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_compress_image_local(tmp_path):
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not installed")
    from litellm.extras.images import compress_image

    p = tmp_path / "t.png"
    im = Image.new("RGB", (32, 32), color=(255, 0, 0))
    im.save(str(p), format="PNG")
    data_url = compress_image(str(p), max_kb=32, cache_dir=tmp_path)
    assert data_url.startswith("data:image/")


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_fetch_remote_image_cache(monkeypatch, tmp_path):
    try:
        import httpx  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("extras not installed")
    from litellm.extras.images import fetch_remote_image

    # create a fake jpeg bytes
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(0, 255, 0)).save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    class _Resp:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, url):
            return _Resp(jpeg_bytes)

    import types
    monkeypatch.setitem(httpx.__dict__, "AsyncClient", _Client)

    async def run():
        return await fetch_remote_image("http://x/y.jpg", cache_dir=tmp_path, timeout=1.0)

    data_url1 = await run()
    data_url2 = await run()
    assert data_url1 == data_url2
    assert data_url1.startswith("data:image/")

