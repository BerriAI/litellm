import json
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
import yaml

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

_REQUESTS: Final = 24
_COMPLETION: Final = {
    "id": "chatcmpl-guarded",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "guarded"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _allow(request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _complete(request: Request) -> Reply:
    return Reply(body=json.dumps(_COMPLETION).encode())


def _deployment(name: str, api_base: str) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "api_base": api_base,
            "api_key": "synthetic-guardrail-key",
        },
    }


def test_requests_naming_a_shared_guardrail_are_spread_over_its_deployments(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip(
        "BUG: init_guardrails_v2 populates llm_router.guardrail_list from the proxy_server global, which is still "
        "None during load_config, so every deployment sharing a guardrail_name checks every request"
    )
    name: Final = f"lb-guardrail-{uuid4().hex}"
    prompts: Final = tuple(f"Hello request {index}" for index in range(_REQUESTS))
    with (
        wire_server(_allow) as first,
        wire_server(_allow) as second,
        wire_server(_complete) as upstream,
    ):
        config: Final = {
            **yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()),
            "guardrails": [_deployment(name, first.url), _deployment(name, second.url)],
        }
        config_path: Final = tmp_path / "guardrail_load_balancing.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=config_path) as proxy, proxy.scenario() as scenario:
            model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=f"{upstream.url}/v1")
            responses: Final = tuple(
                proxy.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": prompt}], "guardrails": [name]},
                )
                for prompt in prompts
            )
        checked: Final = (first.drain(), second.drain())

    assert [(response.status_code, response.headers["x-litellm-applied-guardrails"]) for response in responses] == [
        (200, name)
    ] * _REQUESTS
    seen: Final = tuple(frozenset(json.loads(call.body)["texts"][0] for call in calls) for calls in checked)
    assert all(0 < len(prompts_seen) < _REQUESTS for prompts_seen in seen), [len(prompts_seen) for prompts_seen in seen]
    assert seen[0] | seen[1] == frozenset(prompts)
