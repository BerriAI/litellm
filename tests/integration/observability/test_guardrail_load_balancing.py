import json
from pathlib import Path
from typing import Final
from uuid import uuid4

import yaml

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server


def test_guardrail_response_names_the_applied_guardrail(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"migration-guardrail-{uuid4().hex}"
    prompt: Final = "guardrail migration request"

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        assert json.loads(request.body)["texts"] == [prompt]
        return Reply(body=json.dumps({"action": "NONE"}).encode())

    def upstream(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions"
        body: Final = json.loads(request.body)
        assert body["messages"] == [{"role": "user", "content": prompt}]
        return Reply(
            body=json.dumps(
                {
                    "id": "migration-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "guarded"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ).encode()
        )

    with wire_server(policy) as policy_server, wire_server(upstream) as upstream_server:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy_server.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        config_path: Final = tmp_path / "guardrail.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with owned_proxy(gateway, tmp_path, {}, config=config_path) as proxy, proxy.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_server.url}/v1",
                api_key="synthetic-upstream-key",
            )
            response: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": prompt}], "guardrails": [identity]},
            )

        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "guarded"
        assert response.headers["x-litellm-applied-guardrails"] == identity
        assert len(policy_server.drain()) == 1
        assert len(upstream_server.drain()) == 1
