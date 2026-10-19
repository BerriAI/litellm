import json
from typing import Final, NoReturn

import pytest

import litellm
from tests.test_litellm_rust.ocr.data import OCR_DOCUMENT, OCR_MODEL, ocr_reply
from tests.test_litellm_rust.support.routing import assert_served_natively
from tests._support.provider import SharedProvider

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("caching", [None, False, True])
async def test_ocr_does_not_depend_on_chat_cache(
    server: SharedProvider, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, caching: bool | None
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache(type="local", supported_call_types=["completion", "acompletion"]))
    arguments: Final = {
        "model": OCR_MODEL,
        "document": OCR_DOCUMENT,
        "api_key": "test-key",
        "api_base": server.wire.url,
        "caching": caching,
    }
    server.expect(ocr_reply())
    response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    requests: Final = assert_served_natively(server, response)
    assert response.pages[0].markdown == "native OCR response"
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["invalid", "live", "schema"])
def test_native_projection_errors_never_select_python(
    server: SharedProvider, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import dataclasses
    import ssl

    from litellm.rust_bridge import runtime
    from litellm.rust_bridge.catalog import Route, RouteContext
    from litellm.rust_bridge.host import settings
    from litellm.rust_bridge.ocr.entrypoints import NATIVE_OCR
    from litellm.rust_bridge.public_call import NativeCall

    snapshot: Final = dataclasses.replace(settings.http_settings(), user_agent=1)
    if failure == "schema":
        monkeypatch.setattr(settings, "http_settings", lambda: snapshot)
    else:
        monkeypatch.setattr(
            litellm, "ssl_verify", ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if failure == "live" else object()
        )
    request: Final = NativeCall(
        args=(),
        kwargs={},
        base={
            "model": "mistral/mistral-ocr-latest",
            "document": OCR_DOCUMENT,
            "api_key": "test-key",
            "api_base": server.wire.url,
            "timeout": None,
            "custom_llm_provider": "mistral",
            "extra_headers": None,
        },
    )

    def python_fallback() -> NoReturn:
        pytest.fail("projection failures must not select Python")

    with pytest.raises(RuntimeError if failure == "schema" else ValueError, match="http_settings"):
        runtime.run(
            RouteContext(Route.OCR, provider="mistral"),
            binding=NATIVE_OCR,
            native=lambda native: native(request),
            python=python_fallback,
        )
    requests: Final = server.received()
    assert requests == ()


MODELS: Final = ("cohere/parse-v5.0", "azure_ai/Cohere-parse-v5.0")


IMAGE: Final = {"type": "image_url", "image_url": "data:image/png;base64,YWJj"}


BOX: Final = {"top_left_x": 0, "top_left_y": 0, "bottom_right_x": 32, "bottom_right_y": 32}


PAYLOAD: Final = {
    "pages": [
        {
            "index": 4,
            "markdown": {"content": "receipt", "images": [{"id": "image", "bounding_box": BOX, "description": "scan"}]},
        },
        {"markdown": {"content": "page two"}},
    ],
    "meta": {"billed_units": {"pages": 3}},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_ocr_health_check_uses_the_native_route(recording_server: SharedProvider, model: str) -> None:
    recording_server.expect(ocr_reply(PAYLOAD))
    response: Final = await litellm.ahealth_check(
        model_params={"model": model, "api_key": "test-key", "api_base": recording_server.wire.url}, mode="ocr"
    )
    assert "error" not in response
    requests: Final = assert_served_natively(recording_server)
    body: Final = json.loads(requests[0].body)
    assert body["document"]["image_url"].startswith("data:image/png;base64,")
