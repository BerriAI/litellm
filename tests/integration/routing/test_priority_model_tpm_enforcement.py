import json
import uuid
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

OPENAI_MODEL: Final = "gpt-4.1-mini"
PROMPT_TOKENS: Final = 30
COMPLETION_TOKENS: Final = 10
MODEL_TPM: Final = PROMPT_TOKENS + COMPLETION_TOKENS
PREMIUM_SHARE: Final = 0.5
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_model_tpm_enforcement",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "model tpm control"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
    }
).encode()


@pytest.mark.covers("other.routing.priority_rate_limits.tpm_only_model_rejects_priority_traffic_at_capacity")
def test_tpm_only_model_returns_429_to_priority_key_once_recorded_tokens_reach_model_tpm(
    gateway: Gateway, tmp_path: Path
) -> None:
    probe: Final = "model tpm probe " + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = json.loads(request.body)
        assert body["messages"][0]["content"].startswith(probe), body
        assert body == {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": body["messages"][0]["content"]}],
            "max_tokens": 16,
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
            model=f"openai/{OPENAI_MODEL}",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            tpm=MODEL_TPM,
        )
        key: Final = scenario.key(metadata={"priority": "premium"})
        responses: Final[SimpleQueue[httpx.Response]] = SimpleQueue()

        def attempt() -> httpx.Response:
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"{probe} {uuid.uuid4().hex}"}],
                },
                key=key,
            )
            responses.put(response)
            return response

        first: Final = attempt()
        assert first.status_code == 200, first.text
        assert first.json()["usage"]["total_tokens"] == MODEL_TPM, first.text
        blocked: Final = eventually(attempt, lambda response: response.status_code == 429, seconds=30)
        served: Final = tuple(responses.get_nowait() for _ in range(responses.qsize()))
        assert all(response.status_code == 200 for response in served[:-1]), [r.status_code for r in served]
        assert len(wire.drain()) == len(served) - 1
        assert blocked.headers["x-litellm-priority"] == "premium", blocked.headers
        assert blocked.headers["rate_limit_type"] == "tokens", blocked.headers
        detail: Final = (
            f"Model capacity reached for {model}. Priority: premium, Rate limit type: tokens, "
            f"Model TPM: {MODEL_TPM}, Model RPM: not configured, Remaining: 0"
        )
        assert blocked.json() == {
            "error": {
                "message": detail,
                "type": "throttling_error",
                "param": None,
                "code": "429",
                "provider_specific_fields": {"error": detail},
            }
        }, blocked.text
