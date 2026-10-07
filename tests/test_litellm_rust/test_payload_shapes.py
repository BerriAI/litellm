import asyncio
import json
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import diagnostics
from litellm.litellm_core_utils.payload_shapes import record_shape
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

pytestmark = pytest.mark.requires_rust_extension
_BODY: Final = TypeAdapter(dict[str, JsonValue])
_EVENTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


@pytest.mark.parametrize("asynchronous", (False, True))
def test_real_python_sdk_boundaries_export_shapes_only(monkeypatch, asynchronous):
    monkeypatch.setenv("LITELLM_RUST", "0")
    with recording_service() as collector, recording_service() as provider:
        collector.default_response = ResponseSpec(body={"status": 1})
        provider.default_response = ResponseSpec(body={
            "id": "private-response-id", "object": "chat.completion", "created": 0, "model": "diagnostic-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "private-answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        try:
            assert diagnostics.configure({
                "enabled": True, "payload_shapes": True,
                "policy": {"target_prefixes": ["litellm_payload_shapes"]},
                "destinations": [{"transport": "posthog", "name": "shapes", "api_key": "phc_test", "endpoint": collector.base_url}],
            })
            kwargs: Final = {"messages": [{"role": "user", "content": "private-prompt"}], "api_key": "private-key", "api_base": provider.base_url}
            response: Final = asyncio.run(litellm.acompletion("openai/diagnostic-test", **kwargs)) if asynchronous else litellm.completion("openai/diagnostic-test", **kwargs)
            assert response.choices[0].message.content == "private-answer"
            assert diagnostics.force_flush()
        finally:
            assert diagnostics.shutdown()
        events: Final = _EVENTS.validate_python(_BODY.validate_json(collector.requests[0].raw_body)["batch"])
        fields: Final = tuple(_BODY.validate_python(_BODY.validate_python(event["properties"])["fields"]) for event in events)
        stages: Final = tuple(field["payload.stage"] for field in fields)
        assert stages == ("litellm.request.received", "provider.request.transformed", "provider.request.sent", "provider.response.received", "litellm.response.normalized")
        assert len({field["trace_id"] for field in fields}) == 1
        assert "$['model']" in fields[0]["payload.field_paths"]
        assert "$['messages'][*]['content']" in fields[2]["payload.field_paths"]
        assert "$['choices'][*]['message']['content']" in fields[3]["payload.field_paths"]
        assert all(event["event"] == "llm.payload.shape" for event in events)
        encoded: Final = json.dumps(events)
        assert all(value not in encoded for value in ("private-answer", "private-prompt", "private-response-id", "private-key", provider.base_url))
        assert provider.requests[0].body["messages"][0]["content"] == "private-prompt"


@pytest.mark.parametrize("enabled,sample_rate", ((False, 1), (True, 0)))
def test_capture_opt_in_and_destination_sampling_are_independent(enabled, sample_rate):
    with recording_service() as collector:
        collector.expected_requests = 0
        try:
            assert diagnostics.configure({"enabled": True, "payload_shapes": enabled, "policy": {"sample_rate": sample_rate}, "destinations": [{"transport": "posthog", "name": "shapes", "api_key": "phc_test", "endpoint": collector.base_url}]})
            record_shape("litellm.request.received", {"input": "private"}, "call-id")
            assert diagnostics.force_flush()
        finally:
            assert diagnostics.shutdown()
        assert collector.requests == []
