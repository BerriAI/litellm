"""Regression test for BerriAI/litellm#41421.

Proxy `/v1/images/edits` must forward `image[]`/`mask[]` multipart uploads as
file bytes, never as `str()`-serialized objects. Previously `_read_request_body`
left the collapsed `image[]` UploadFile in `data` alongside the converted
`image` BytesIO list, and downstream `flatten_form_field_values` stringified
the leftover into a form field OpenAI rejected with
"Invalid type for 'image[0]': expected a file, but got a string instead."
"""

import io
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.image_endpoints import endpoints

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def _client(monkeypatch, captured: dict[str, Any]) -> TestClient:
    class CaptureProcessing:
        def __init__(self, data: dict[str, Any]) -> None:
            captured.update(data)

        async def base_process_llm_request(self, **_: Any) -> dict[str, Any]:
            return {"data": [{"b64_json": "aGk="}]}

    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", CaptureProcessing)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)

    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth()
    return TestClient(app)


def test_image_bracket_upload_stays_bytes_not_str(monkeypatch):
    """`image[]` file must reach the provider as BytesIO with no `image[]` leftover."""
    captured: dict[str, Any] = {}
    response = _client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={"image[]": ("input.png", PNG_BYTES, "image/png")},
        data={"model": "openai/gpt-image-2", "prompt": "Make this image photorealistic"},
    )

    assert response.status_code == 200
    # No raw multipart key may leak downstream where flatten would str() it.
    assert "image[]" not in captured
    assert "mask[]" not in captured
    images = captured["image"]
    assert isinstance(images, list) and len(images) == 1
    assert isinstance(images[0], io.BytesIO)
    assert not isinstance(images[0], str)
    assert not any(isinstance(v, str) and "BytesIO object at" in v for v in images)
    images[0].seek(0)
    assert images[0].read() == PNG_BYTES
    assert getattr(images[0], "name", None) == "input.png"


def test_mask_bracket_upload_stays_bytes_not_str(monkeypatch):
    """`mask[]` file must reach the provider as BytesIO with no `mask[]` leftover."""
    captured: dict[str, Any] = {}
    response = _client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={
            "image": ("input.png", PNG_BYTES, "image/png"),
            "mask[]": ("mask.png", PNG_BYTES, "image/png"),
        },
        data={"model": "openai/gpt-image-2", "prompt": "hi"},
    )

    assert response.status_code == 200
    assert "image[]" not in captured
    assert "mask[]" not in captured
    masks = captured["mask"]
    assert isinstance(masks, list) and len(masks) == 1
    assert isinstance(masks[0], io.BytesIO)
    masks[0].seek(0)
    assert masks[0].read() == PNG_BYTES


def test_multiple_bracket_uploads_all_reach_provider(monkeypatch):
    """Repeated `image[]` parts must all arrive (form-dict collapsing must not drop any)."""
    captured: dict[str, Any] = {}
    client = _client(monkeypatch, captured)
    png2 = b"\x89PNG\r\n\x1a\n" + b"\x01" * 100
    response = client.post(
        "/v1/images/edits",
        files=[
            ("image[]", ("a.png", PNG_BYTES, "image/png")),
            ("image[]", ("b.png", png2, "image/png")),
        ],
        data={"model": "openai/gpt-image-2", "prompt": "hi"},
    )

    assert response.status_code == 200
    assert "image[]" not in captured
    assert len(captured["image"]) == 2
    assert [getattr(b, "name", None) for b in captured["image"]] == ["a.png", "b.png"]


def test_string_image_upload_rejected(monkeypatch):
    """A string `image`/`image[]` field must 422, not forward as a string file."""
    for field in ("image", "image[]"):
        captured: dict[str, Any] = {}
        response = _client(monkeypatch, captured).post(
            "/v1/images/edits",
            data={"model": "openai/gpt-image-2", "prompt": "hi", field: "not-a-file"},
        )
        assert response.status_code == 422, field


def test_numeric_coercion_preserved_with_file_upload(monkeypatch):
    """Numeric coercion (`n` -> int) must keep working alongside the file fix."""
    captured: dict[str, Any] = {}
    response = _client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={"image[]": ("input.png", PNG_BYTES, "image/png")},
        data={"model": "openai/gpt-image-2", "prompt": "hi", "n": "2"},
    )

    assert response.status_code == 200
    assert captured["n"] == 2 and isinstance(captured["n"], int)
    assert isinstance(captured["image"][0], io.BytesIO)


def test_transformation_preserves_bytesio_filename():
    """BytesIO `.name` set by the proxy must survive into the httpx files tuple."""
    from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
    from litellm.types.router import GenericLiteLLMParams

    buf = io.BytesIO(PNG_BYTES)
    buf.name = "input.png"
    data, files = OpenAIImageEditConfig().transform_image_edit_request(
        model="gpt-image-2",
        prompt="hi",
        image=[buf],
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "image" not in data  # files stay out of the data dict
    assert files[0][0] == "image[]"
    assert files[0][1][0] == "input.png"
    assert files[0][1][1] is buf
    assert isinstance(files[0][1][1], io.BytesIO)
