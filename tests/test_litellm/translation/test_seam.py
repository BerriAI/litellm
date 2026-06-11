"""The completion() fork: flag-gated v2 execution with fail-closed fallback.

Uses respx to intercept the anthropic endpoint, so these run the REAL
pipeline (translate -> http port -> response translate -> ModelResponse)
end to end without network.
"""

import json

import pytest
import respx
from httpx import Response

import litellm

_URL = "https://api.anthropic.com/v1/messages"

_PROVIDER_RESPONSE = {
    "id": "msg_seam_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "Hello from v2."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


@pytest.fixture(autouse=True)
def _flag(monkeypatch):
    monkeypatch.setattr(litellm, "translation_v2_providers", ["anthropic"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@respx.mock
def test_flag_on_serves_through_v2() -> None:
    route = respx.post(_URL).mock(return_value=Response(200, json=_PROVIDER_RESPONSE))
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert route.called
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "claude-sonnet-4-5"
    assert "json_mode" not in sent
    assert response.choices[0].message.content == "Hello from v2."
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5


@respx.mock
def test_unsupported_shape_falls_back_to_v1() -> None:
    route = respx.post(_URL).mock(return_value=Response(200, json=_PROVIDER_RESPONSE))
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        max_tokens=32,
        messages=[
            {"role": "user", "content": "hi"},
            # assistant prefix is a v1 feature outside v2's surface: the
            # request must be served (by v1), never rejected or dropped.
            {"role": "assistant", "content": "Hel", "prefix": True},
        ],
    )
    assert route.called
    assert response.choices[0].message.content == "Hello from v2."


@respx.mock
def test_provider_error_raises_contract_not_fallback() -> None:
    route = respx.post(_URL).mock(
        return_value=Response(429, json={"error": {"message": "rate limited"}})
    )
    with pytest.raises(litellm.exceptions.RateLimitError):
        litellm.completion(
            model="anthropic/claude-sonnet-4-5",
            max_tokens=32,
            messages=[{"role": "user", "content": "hi"}],
            num_retries=0,
        )
    assert route.call_count == 1  # sent once: no silent re-send through v1


@pytest.mark.asyncio
@respx.mock
async def test_acompletion_path_returns_awaitable() -> None:
    respx.post(_URL).mock(return_value=Response(200, json=_PROVIDER_RESPONSE))
    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response.choices[0].message.content == "Hello from v2."


def test_flag_off_returns_none_from_seam() -> None:
    from litellm.translation_seam import try_completion_v2

    litellm.translation_v2_providers = []
    assert (
        try_completion_v2(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_param_args={},
            non_default_params={},
            api_key=None,
            api_base=None,
            timeout=None,
            stream=None,
            acompletion=None,
            logging_obj=None,
            model_response=None,
            request_drop_params=None,
        )
        is None
    )


def test_streaming_stays_on_v1() -> None:
    from litellm.translation_seam import try_completion_v2

    assert (
        try_completion_v2(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_param_args={},
            non_default_params={},
            api_key=None,
            api_base=None,
            timeout=None,
            stream=True,
            acompletion=None,
            logging_obj=None,
            model_response=None,
            request_drop_params=None,
        )
        is None
    )
