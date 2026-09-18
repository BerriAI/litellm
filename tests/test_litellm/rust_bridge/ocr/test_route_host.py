from collections.abc import Generator
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import bindings
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest
from litellm.rust_bridge.ocr.route_host import NativeFailure, RequestFailure, UpstreamFailure, map_failure
from litellm.rust_bridge.ocr.route_host import response as build_ocr_response

REQUEST: Final = LiteLLMOcrRequest(
    model="mistral/mistral-ocr-latest",
    document={"type": "document_url", "document_url": "https://example.com/file.pdf"},
    api_key="test-key",
    api_base=None,
    timeout=None,
    custom_llm_provider=None,
    extra_headers=None,
    kwargs={"req_format": "markdown"},
)


class RustBridgeDeclined(Exception):
    pass


class RustUpstreamError(Exception):
    pass


@pytest.fixture(autouse=True)
def native_exceptions(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    native: Final = SimpleNamespace(RustBridgeDeclined=RustBridgeDeclined, RustUpstreamError=RustUpstreamError)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    yield


def test_rust_ocr_response_retains_provider_native_response():
    provider_response = {"status": "succeeded", "analyzeResult": {"content": "native"}}
    response = build_ocr_response(
        {
            "pages": [],
            "model": "prebuilt-layout",
            "document_annotation": None,
            "usage_info": {"pages_processed": 0},
            "object": "ocr",
            "provider_native_response": provider_response,
        }
    )

    assert response.get_provider_native_response() == provider_response
    assert response.model_dump().get("provider_native_response") is None


def test_map_failure_builds_public_error_from_upstream_status_and_headers() -> None:
    error: Final = RustUpstreamError(429, '{"message": "slow down"}', (("retry-after", "7"),))

    public_error: Final = map_failure(error, REQUEST, "mistral")

    assert isinstance(public_error, litellm.RateLimitError)
    assert public_error.status_code == 429
    assert public_error.response.headers["retry-after"] == "7"
    assert public_error.response.text == '{"message": "slow down"}'
    assert public_error.__context__ is error
    assert public_error.llm_provider == "mistral"


def test_map_failure_reports_a_missing_response_as_a_connection_error() -> None:
    error: Final = RustUpstreamError(0, "upstream network error: timed out", ())

    public_error: Final = map_failure(error, REQUEST, "mistral")

    assert isinstance(public_error, litellm.APIConnectionError)
    assert "timed out" in str(public_error)
    assert not isinstance(public_error.__context__, UpstreamFailure)
    assert public_error.__context__ is error


def test_map_failure_turns_request_rejections_into_bad_requests() -> None:
    error: Final = RustBridgeDeclined("Document URL is required", "invalid_request")

    public_error: Final = map_failure(error, REQUEST, "mistral")

    assert isinstance(public_error, litellm.BadRequestError)
    assert public_error.status_code == 400
    assert "Document URL is required" in str(public_error)
    assert not isinstance(public_error.__context__, RequestFailure)
    assert public_error.__context__ is error


def test_map_failure_keeps_other_rejections_status_free() -> None:
    error: Final = RustBridgeDeclined("Missing Azure AI credentials", "credential")

    public_error: Final = map_failure(error, REQUEST, "mistral")

    assert isinstance(public_error, litellm.APIConnectionError)
    assert "Missing Azure AI credentials" in str(public_error)
    assert not isinstance(public_error.__context__, NativeFailure)
    assert public_error.__context__ is error


def test_map_failure_leaves_non_native_errors_unwrapped() -> None:
    error: Final = RuntimeError("bridge exploded")

    public_error: Final = map_failure(error, REQUEST, "mistral")

    assert isinstance(public_error, litellm.APIConnectionError)
    assert "bridge exploded" in str(public_error)


def test_map_failure_reports_invalid_request_format_as_unsupported_params() -> None:
    error: Final = RustBridgeDeclined("Invalid `req_format`.", "request_format")

    with pytest.raises(litellm.UnsupportedParamsError, match="Invalid `req_format`: 'markdown'"):
        raise map_failure(error, REQUEST, "mistral")
