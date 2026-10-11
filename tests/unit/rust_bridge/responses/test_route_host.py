import datetime
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge.responses.route_host import (
    connection_defaults,
    map_failure,
    response,
    stream_decoder,
    stream_hidden_params,
)
from litellm.types.llms.openai import OutputTextDeltaEvent, ResponseCompletedEvent, ResponsesAPIResponse


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


def test_stream_hidden_params_project_upstream_headers_the_way_the_python_iterator_does() -> None:
    hidden: Final = stream_hidden_params((("x-request-id", "req_upstream"), ("content-type", "text/event-stream")))

    additional: Final = hidden["additional_headers"]
    assert isinstance(additional, dict)
    assert additional["llm_provider-x-request-id"] == "req_upstream"
    assert all(isinstance(value, str) for value in additional.values())


def _request(**overrides: object) -> dict[str, object]:
    logger: Final = Logging(
        model="gpt-4.1",
        messages=[],
        stream=True,
        call_type="aresponses",
        start_time=datetime.datetime.now(),
        litellm_call_id="stream",
        function_id="stream",
    )
    return {"model": "openai/gpt-4.1", "input": "hi", "litellm_logging_obj": logger, **overrides}


def test_stream_decoder_types_events_with_the_provider_config() -> None:
    decode: Final = stream_decoder(_request())

    delta: Final = decode(
        {"type": "response.output_text.delta", "item_id": "msg", "output_index": 0, "content_index": 0, "delta": "hi"}
    )
    completed: Final = decode(
        {
            "type": "response.completed",
            "response": {"id": "resp", "object": "response", "created_at": 1, "model": "gpt-4.1", "output": []},
        }
    )

    assert isinstance(delta, OutputTextDeltaEvent)
    assert delta.delta == "hi"
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "resp"


def test_stream_decoder_rejects_a_request_without_the_injected_logger() -> None:
    with pytest.raises(TypeError):
        stream_decoder({"model": "openai/gpt-4.1", "litellm_logging_obj": None})
