import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_ADVISOR_KEY: Final = "synthetic-advisor-key"
_QUESTION: Final = "which index should this query use"
_PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"


def _executor_reply(body: dict[str, object], identity: str) -> Reply:
    tools: Final = body.get("tools")
    message: Final = (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "advisor-call",
                    "type": "function",
                    "function": {"name": "advisor", "arguments": json.dumps({"question": _QUESTION})},
                }
            ],
        }
        if isinstance(tools, list)
        else {"role": "assistant", "content": "served without an advisor"}
    )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{identity}-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": 1,
                "model": "llama-3.3-70b-versatile",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tools else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        ).encode()
    )


def _cooldowns_enabled_config(directory: Path) -> Path:
    loaded: Final = yaml.safe_load(_PROXY_CONFIG.read_text())
    path: Final = directory / "cooldowns_enabled.yaml"
    path.write_text(yaml.safe_dump({**loaded, "router_settings": {"num_retries": 0}}))
    return path


@pytest.mark.covers("routing.cooldown.advisor_sub_call_failure_does_not_cool_down_the_executor_deployment")
def test_advisor_sub_call_401_leaves_the_executor_deployment_serving_the_next_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "advisor-cooldown-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.target == "/v1/chat/completions":
            return _executor_reply(json.loads(request.body), identity)
        assert request.target == "/v1/messages"
        assert request.headers["x-api-key"] == _ADVISOR_KEY
        return Reply(
            status=401,
            body=json.dumps(
                {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
            ).encode(),
        )

    with (
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_cooldowns_enabled_config(tmp_path)) as candidate,
        candidate.scenario() as scenario,
    ):
        executor: Final = scenario.model(model="hosted_vllm/gpt-4o-mini", api_base=wire.url + "/v1")
        advisor: Final = scenario.model(
            model="anthropic/claude-opus-4-1-20250805", api_base=wire.url, api_key=_ADVISOR_KEY
        )
        advised: Final = candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": executor,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": identity}],
                "tools": [{"type": "advisor_20260301", "name": "advisor", "model": advisor}],
            },
        )
        assert advised.status_code == 401, advised.text
        assert [request.target for request in wire.drain()] == ["/v1/chat/completions", "/v1/messages"]
        unrelated: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": executor, "messages": [{"role": "user", "content": identity + " unrelated"}]},
        )
        assert unrelated.status_code == 200, unrelated.text
        assert unrelated.json()["choices"][0]["message"]["content"] == "served without an advisor", unrelated.text
        assert [request.target for request in wire.drain()] == ["/v1/chat/completions"]
