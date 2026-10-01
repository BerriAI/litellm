import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
MODEL_RPM: Final = 40
MODEL_TPM: Final = 1000
PREMIUM_SHARE: Final = 0.5
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "msg_priority_headers",
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "priority header control"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
).encode()


CHAT_MODEL: Final = "gpt-5.6"
MAX_COMPLETION_TOKENS: Final = 64


def _chat_frames(identity: str, text: str) -> tuple[bytes, ...]:
    events: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
    )
    frames: Final = tuple(
        b"data: "
        + json.dumps(
            {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": CHAT_MODEL, **event}
        ).encode()
        + b"\n\n"
        for event in events
    )
    return (*frames, b"data: [DONE]\n\n")


@pytest.mark.covers("other.routing.priority_rate_limits.v1_messages_success_exposes_v3_priority_headers")
def test_non_streaming_v1_messages_success_carries_v3_priority_rate_limit_headers(
    gateway: Gateway, tmp_path: Path
) -> None:
    probe: Final = "priority header probe " + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        assert json.loads(request.body) == {
            "model": ANTHROPIC_MODEL,
            "messages": [{"role": "user", "content": probe}],
            "max_tokens": 16,
            "stream": False,
        }
        return Reply(body=UPSTREAM_REPLY)

    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3"],
        "priority_reservation": {"premium": PREMIUM_SHARE},
    }
    path: Final = tmp_path / "priority.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            rpm=MODEL_RPM,
            tpm=MODEL_TPM,
        )
        key: Final = scenario.key(metadata={"priority": "premium"})
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": probe}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": "priority header control"}], response.text
        assert len(wire.drain()) == 1
        expected: Final = {
            "x-litellm-priority": "premium",
            "x-litellm-rate-limiter-version": "v3",
            "x-ratelimit-model_saturation_check-limit-requests": str(MODEL_RPM),
            "x-ratelimit-model_saturation_check-remaining-requests": str(MODEL_RPM - 1),
            "x-ratelimit-priority_model-limit-requests": str(int(MODEL_RPM * PREMIUM_SHARE)),
            "x-ratelimit-priority_model-remaining-requests": str(int(MODEL_RPM * PREMIUM_SHARE) - 1),
            "x-ratelimit-priority_model-limit-tokens": str(int(MODEL_TPM * PREMIUM_SHARE)),
            "x-ratelimit-priority_model-remaining-tokens": str(int(MODEL_TPM * PREMIUM_SHARE) - 1),
        }
        observed: Final = {name: response.headers.get(name) for name in expected}
        assert observed == expected, response.headers


@pytest.mark.covers("other.routing.priority_rate_limits.streaming_success_logs_v3_remaining_values_for_callbacks")
def test_streaming_chat_completion_success_logs_v3_rate_limit_remaining_values_for_callbacks(
    gateway: Gateway, tmp_path: Path
) -> None:
    probe: Final = "streaming remaining probe " + uuid.uuid4().hex
    sink_secret: Final = "synthetic-sink-secret-" + uuid.uuid4().hex

    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert json.loads(request.body) == {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": probe}],
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }, request.body
        return Reply(content_type="text/event-stream", chunks=_chat_frames("chatcmpl_" + probe[-8:], "streamed"))

    def sink(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {sink_secret}"
        return Reply()

    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["generic_api"],
        "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
    }
    path: Final = tmp_path / "per_key_streaming.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        wire_server(provider) as wire,
        wire_server(sink) as endpoint,
        owned_proxy(
            gateway,
            tmp_path,
            {"GENERIC_LOGGER_ENDPOINT": endpoint.url, "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {sink_secret}"},
            config=path,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"openai/{CHAT_MODEL}",
            api_base=wire.url + "/v1",
            api_key="synthetic-openai-key",
        )
        key: Final = scenario.key(model_rpm_limit={model: MODEL_RPM}, model_tpm_limit={model: MODEL_TPM})
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": probe}],
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        assert '"content":"streamed"' in response.text, response.text
        assert len(wire.drain()) == 1
        batches: Final[
            list[Request]
        ] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

        def delivered() -> tuple[dict, ...]:
            batches.extend(endpoint.drain())
            return tuple(
                event for batch in batches for event in json.loads(batch.body) if event.get("model_group") == model
            )

        events: Final = eventually(delivered, lambda values: len(values) == 1, seconds=10)
        assert (events[0]["status"], events[0]["stream"]) == ("success", True), json.dumps(events[0])
        additional_headers: Final = events[0]["hidden_params"]["additional_headers"] or {}
        observed: Final = {name: value for name, value in additional_headers.items() if name.startswith("x-ratelimit-")}
        remaining_tokens: Final = observed.get("x-ratelimit-model_per_key-remaining-tokens")
        assert isinstance(remaining_tokens, int) and 0 < remaining_tokens <= MODEL_TPM, json.dumps(observed)
        assert {name: value for name, value in observed.items() if not name.endswith("-remaining-tokens")} == {
            "x-ratelimit-model_per_key-limit-requests": MODEL_RPM,
            "x-ratelimit-model_per_key-remaining-requests": MODEL_RPM - 1,
            "x-ratelimit-model_per_key-limit-tokens": MODEL_TPM,
        }, json.dumps(events[0]["hidden_params"])
