import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
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
