from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.responses.route_host import arguments, connection_defaults, response, unsupported_request
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


def test_arguments_preserve_the_bound_view() -> None:
    kwargs: Final = MappingProxyType({"litellm_metadata": {"user_id": "u"}})
    request: Final = NativeCall(
        args=(),
        kwargs=kwargs,
        bound={
            "model": "gpt-4o",
            "input": "hi",
            "stream": None,
            "api_key": None,
            "api_base": None,
            "custom_llm_provider": "openai",
            "extra_headers": None,
            **kwargs,
        },
    )

    assert arguments(request.bound) is request.bound


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


@pytest.mark.parametrize(
    ("fields", "expected"),
    (
        ({"model": "gpt-4o", "input": "hi"}, None),
        ({"model": "openai/gpt-4o", "input": "hi"}, None),
        ({"model": "gpt-4o", "input": "hi", "custom_llm_provider": "openai"}, None),
        ({"model": "claude-sonnet-4-5", "input": "hi"}, "native HTTP responses provider"),
        ({"model": "anthropic/claude-sonnet-4-5", "input": "hi"}, "native HTTP responses provider"),
        ({"model": "gpt-4o", "input": "hi", "custom_llm_provider": "azure"}, "native HTTP responses provider"),
        ({"model": "openai/team/gpt-4o", "input": "hi"}, "native HTTP responses provider"),
        ({"model": "not-a-known-model-xyz", "input": "hi"}, "native Responses could not resolve the provider"),
        ({"model": "gpt-4o", "input": "hi", "mock_response": "x"}, "native inference does not implement mock_response"),
    ),
)
def test_unsupported_request_names_what_the_native_route_cannot_serve(
    fields: dict[str, object], expected: str | None
) -> None:
    assert unsupported_request(fields) == expected
