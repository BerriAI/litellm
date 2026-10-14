from typing import Final, NoReturn

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.ocr.data import OCR_DOCUMENT, OCR_MODEL
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.parametrize("enabled", [False, True, None])
def test_public_ocr_uses_native_route_independently_of_flag(server: RecordingServer, enabled: bool | None) -> None:
    litellm.rust(enabled)
    response: Final = litellm.ocr(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key="test-key",
        api_base=server.base_url,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "native OCR response"
    assert_served_natively(server)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("caching", [None, False, True])
async def test_ocr_does_not_depend_on_chat_cache(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, caching: bool | None
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache(type="local", supported_call_types=["completion", "acompletion"]))
    arguments: Final = {
        "model": OCR_MODEL,
        "document": OCR_DOCUMENT,
        "api_key": "test-key",
        "api_base": server.base_url,
        "caching": caching,
    }
    response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert response.pages[0].markdown == "native OCR response"
    assert len(server.requests) == 1


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("failure", ["invalid", "live", "schema"])
def test_native_projection_errors_never_select_python(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, required: bool, failure: str
) -> None:
    import dataclasses
    import ssl

    from litellm.rust_bridge import runtime
    from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
    from litellm.rust_bridge.configuration import Rollout
    from litellm.rust_bridge.host import settings
    from litellm.rust_bridge.ocr.entrypoints import NATIVE_OCR
    from litellm.rust_bridge.public_call import NativeCall

    server.expected_requests = 0
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
            "api_base": server.base_url,
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
            rules=(RouteRule(Route.OCR, Rollout.RUST_REQUIRED if required else Rollout.RUST_OPT_OUT),),
        )
    assert server.requests == []


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
async def test_ocr_health_check_uses_the_native_route(recording_server: RecordingServer, model: str) -> None:
    recording_server.enqueue(ResponseSpec(body=PAYLOAD))
    response: Final = await litellm.ahealth_check(
        model_params={"model": model, "api_key": "test-key", "api_base": recording_server.base_url}, mode="ocr"
    )
    assert "error" not in response
    assert_served_natively(recording_server)
    assert recording_server.requests[0].body["document"]["image_url"].startswith("data:image/png;base64,")
