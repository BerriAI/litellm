from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.rust_bridge.responses.route_host import connection_defaults, map_failure, response
from litellm.types.llms.openai import ResponsesAPIResponse


def test_response_validates_into_the_public_responses_model() -> None:
    built: Final = response(
        MappingProxyType(
            {
                "id": "resp_native",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_native",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "native", "annotations": []}],
                    }
                ],
            }
        )
    )

    assert isinstance(built, ResponsesAPIResponse)
    assert built.id == "resp_native"
    assert built.output[0].content[0].text == "native"


def test_response_rejects_a_payload_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        response(MappingProxyType({"object": "response"}))


@pytest.mark.parametrize(
    ("global_key", "provider_key", "expected"),
    (
        ("global", "provider", "global"),
        (None, "provider", "provider"),
        ("", "provider", "provider"),
        (None, None, None),
    ),
)
def test_connection_defaults_preserve_openai_precedence(
    monkeypatch: pytest.MonkeyPatch, global_key: str | None, provider_key: str | None, expected: str | None
) -> None:
    monkeypatch.setattr(litellm, "api_key", global_key)
    monkeypatch.setattr(litellm, "openai_key", provider_key)
    monkeypatch.setattr(litellm, "api_base", "https://configured.invalid/v1")
    assert connection_defaults("openai") == (expected, litellm.api_base)


class _UpstreamFailure(Exception):
    headers: Final = ()


@pytest.mark.parametrize(
    ("api_base", "base_url", "expected"),
    (
        (None, "https://alias.invalid/v1", "https://alias.invalid/v1"),
        ("", "https://alias.invalid/v1", "https://alias.invalid/v1"),
        ("https://base.invalid/v1", "https://alias.invalid/v1", "https://base.invalid/v1"),
    ),
)
def test_failure_preserves_the_explicit_endpoint(api_base: str | None, base_url: str, expected: str) -> None:
    upstream: Final = _UpstreamFailure(429, '{"error":{"message":"rate limited"}}')
    mapped: Final = map_failure(upstream, {"model": "openai/test-model", "api_base": api_base, "base_url": base_url})
    assert isinstance(mapped, litellm.RateLimitError)
    assert str(mapped.response.request.url) == expected
    assert mapped.__context__ is upstream
