from dataclasses import dataclass, field
from typing import Final

from litellm.rust_bridge.callback_adapters import ProviderLoggingAdapter, SessionCallbackAdapter
from litellm.rust_bridge.callbacks import CallbackDecision


@dataclass
class RecordingLogging:
    model_call_details: dict[str, object] = field(default_factory=dict)
    calls: list[tuple[str, object]] = field(default_factory=list)

    def pre_call(self, *, input: object, api_key: str | None, additional_args: object) -> None:
        self.calls.append(("pre", (input, api_key, additional_args)))

    def post_call(self, *, original_response: str, input: object, api_key: str | None) -> None:
        self.calls.append(("post", (original_response, input, api_key)))


def provider_event(**updates: object) -> dict[str, object]:
    return {
        "provider": "mistral",
        "model": "mistral-ocr-latest",
        "call_id": "call-1",
        "trace_id": "trace-1",
        "attempt": 1,
        "started_at": 10.0,
        **updates,
    }


def test_provider_logging_adapter_preserves_provider_lifecycle() -> None:
    logging: Final = RecordingLogging()
    adapter: Final = ProviderLoggingAdapter(logging, "OCR document processing", "secret")

    assert adapter.pre_call(
        provider_event(request={"model": "mistral-ocr-latest"}, api_base="https://provider.test", headers={})
    ) == {"action": "unchanged"}
    assert adapter.post_call(provider_event(response={"pages": []}, status_code=200, headers={}, ended_at=11.0)) == {
        "action": "unchanged"
    }
    adapter.error(
        provider_event(
            message="retryable",
            stage="provider_response",
            committed=True,
            status_code=429,
            will_retry=True,
            ended_at=11.0,
        )
    )
    assert adapter.stream_event(provider_event(event={"type": "delta"}, sequence=1)) == {"action": "unchanged"}
    adapter.stream_close(provider_event(outcome="completed", ended_at=12.0))

    assert [name for name, _ in logging.calls] == ["pre", "post"]
    assert logging.model_call_details["provider_stream_event"] == {
        **provider_event(),
        "event": {"type": "delta"},
        "sequence": 1,
    }
    assert logging.model_call_details["provider_stream_close"] == {
        **provider_event(),
        "outcome": "completed",
        "ended_at": 12.0,
    }
    assert logging.model_call_details["provider_error"] == {
        **provider_event(),
        "message": "retryable",
        "stage": "provider_response",
        "committed": True,
        "status_code": 429,
        "will_retry": True,
        "ended_at": 11.0,
    }


@dataclass
class RecordingSession:
    events: list[str] = field(default_factory=list)

    def before_connect(self, payload: object, /) -> CallbackDecision:
        self.events.append("before_connect")
        return {"action": "unchanged"}

    def connected(self, payload: object, /) -> None:
        self.events.append("connected")

    def before_send(self, payload: object, /) -> CallbackDecision:
        self.events.append("before_send")
        return {"action": "reject", "message": "drop frame", "status_code": None}

    def after_receive(self, payload: object, /) -> CallbackDecision:
        self.events.append("after_receive")
        return {"action": "replace", "payload": {"type": "masked"}}

    def response_complete(self, payload: object, /) -> None:
        self.events.append("response_complete")

    def response_error(self, payload: object, /) -> None:
        self.events.append("response_error")

    def error(self, payload: object, /) -> None:
        self.events.append("error")

    def close(self, payload: object, /) -> None:
        self.events.append("close")


def test_session_adapter_preserves_frame_decisions_and_order() -> None:
    callback: Final = RecordingSession()
    adapter: Final = SessionCallbackAdapter(callback)
    event: Final = {"session_id": "session-1", "call_id": "call-1", "event": {"type": "response.create"}}

    assert adapter.before_connect(event) == {"action": "unchanged"}
    adapter.connected(event)
    assert adapter.before_send(event) == {"action": "reject", "message": "drop frame", "status_code": None}
    assert adapter.after_receive(event) == {"action": "replace", "payload": {"type": "masked"}}
    adapter.response_complete(event)
    adapter.response_error(event)
    adapter.error(event)
    adapter.close(event)

    assert callback.events == [
        "before_connect",
        "connected",
        "before_send",
        "after_receive",
        "response_complete",
        "response_error",
        "error",
        "close",
    ]
