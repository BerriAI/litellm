from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge import responses as bridge

RUST_RESPONSE: Final[Mapping[str, object]] = {
    "id": "resp_test",
    "created_at": 123,
    "model": "gpt-5",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "id": "msg_test",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello", "annotations": list[object]()}],
        }
    ],
}


class _RecordingGate:
    def __init__(self, reason: str | None = None) -> None:
        self.reason: Final = reason
        self.calls: list[Mapping[str, object]] = []

    def __call__(
        self,
        *,
        model: str,
        input: str | Sequence[object],
        optional_params: Mapping[str, object] | None,
        custom_llm_provider: str | None,
        use_chat_completions_api: bool,
    ) -> str | None:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "optional_params": optional_params,
                "custom_llm_provider": custom_llm_provider,
                "use_chat_completions_api": use_chat_completions_api,
            }
        )
        return self.reason


class _RecordingCall:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    def __call__(
        self,
        *,
        model: str,
        input: str | Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        use_chat_completions_api: bool | None,
    ) -> Mapping[str, object]:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "optional_params": optional_params,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "timeout_seconds": timeout_seconds,
                "use_chat_completions_api": use_chat_completions_api,
            }
        )
        return RUST_RESPONSE


@pytest.fixture(autouse=True)
def reset_bridge() -> Iterator[None]:
    bridge.set_rust_responses(responses=None, decline=None)
    configuration.reset_rust_configuration()
    yield
    bridge.set_rust_responses(responses=None, decline=None)
    configuration.reset_rust_configuration()


def _accepts() -> bool:
    return bridge.rust_responses_accepts(
        model="gpt-5",
        input="hello",
        optional_params={"max_output_tokens": 16},
        custom_llm_provider="openai",
        use_chat_completions_api=False,
        stream=None,
        request_override=None,
        extra_body=None,
        extra_query=None,
    )


def test_disabled_route_does_not_consult_native_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    gate: Final = _RecordingGate()
    bridge.set_rust_responses(decline=gate)
    assert _accepts() is False
    assert gate.calls == []


def test_enabled_route_uses_native_responses_abstraction() -> None:
    gate: Final = _RecordingGate()
    call: Final = _RecordingCall()
    bridge.set_rust_responses(responses=call, decline=gate)
    configuration.use_litellm_rust(True)
    assert _accepts() is True
    response: Final = bridge.responses(
        model="gpt-5",
        input="hello",
        optional_params={"max_output_tokens": 16},
        api_key="test-key",
        api_base="http://provider.test",
        custom_llm_provider="openai",
        extra_headers=None,
        timeout=5,
        use_chat_completions_api=False,
    )
    assert response is not None
    assert response.output_text == "hello"
    assert call.calls[0]["use_chat_completions_api"] is False


def test_unavailable_bridge_stays_on_python(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration.use_litellm_rust(True)
    monkeypatch.setattr(bridge, "get_native_bridge", lambda: None)
    assert _accepts() is False
