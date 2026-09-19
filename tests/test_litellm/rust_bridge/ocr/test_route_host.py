from typing import Final

import pytest

import litellm
from litellm.rust_bridge.ocr.route_host import UpstreamFailure, map_failure
from litellm.rust_bridge.ocr.route_host import response as build_ocr_response
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest

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


class RustUpstreamError(Exception):
    def __init__(self, status: int, body: str, headers: tuple[tuple[str, str], ...]) -> None:
        super().__init__(status, body)
        self.headers: Final = list(headers)


class RustFormatError(Exception):
    ocr_request_format_error: Final = True


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


def test_map_failure_maps_upstream_401_to_authentication_error() -> None:
    error: Final = RustUpstreamError(401, '{"message": "Unauthorized"}', ())

    public_error: Final = map_failure(error, REQUEST, "mistral")

    assert isinstance(public_error, litellm.AuthenticationError)
    assert public_error.status_code == 401
    assert public_error.response.text == '{"message": "Unauthorized"}'
    assert public_error.__context__ is error


def test_map_failure_leaves_non_upstream_errors_unwrapped() -> None:
    error: Final = RuntimeError("bridge exploded")

    public_error: Final = map_failure(error, REQUEST, "mistral")

    assert not isinstance(public_error, UpstreamFailure)
    assert isinstance(public_error, litellm.APIConnectionError)
    assert "bridge exploded" in str(public_error)


def test_map_failure_reports_invalid_request_format_as_unsupported_params() -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="Invalid `req_format`: 'markdown'"):
        raise map_failure(RustFormatError(), REQUEST, "mistral")


def test_failure_before_projection_reports_the_provider_python_reports() -> None:
    error: Final = ValueError("Document URL is required")
    error.status_code = 400  # type: ignore[attr-defined]

    mapped: Final = map_failure(error, REQUEST, "")

    assert isinstance(mapped, litellm.BadRequestError)
    assert mapped.llm_provider == "mistral"
    assert "BadRequestError:  - " not in str(mapped)


@pytest.mark.parametrize(
    ("provider", "expected_type", "expected_message"),
    [
        ("openai", litellm.APIConnectionError, "OCR is not supported for provider: openai"),
        ("not_a_provider", litellm.BadRequestError, "LLM Provider NOT provided"),
    ],
)
def test_invalid_provider_failures_follow_the_python_route(
    provider: str, expected_type: type[Exception], expected_message: str
) -> None:
    error: Final = ValueError(f"invalid provider: {provider}")
    error.ocr_invalid_provider = provider  # type: ignore[attr-defined]

    mapped: Final = map_failure(error, REQUEST, "")

    assert isinstance(mapped, expected_type)
    assert expected_message in str(mapped)
