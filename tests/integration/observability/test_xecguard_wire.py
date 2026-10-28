import json
import uuid
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def test_xecguard_post_call_scan_reaches_vendor_and_call_succeeds(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "xecguard" + uuid.uuid4().hex

    def vendor(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/xecguard/v1/scan"
        assert request.headers["authorization"] == "Bearer synthetic-xecguard-key"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "xecguard_v2"
        assert body["scan_type"] in ("input", "response")
        assert any(message.get("content") == "hi" for message in body.get("messages", [])), body
        return Reply(body=json.dumps({"decision": "SAFE", "violations": []}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/chat/completions"
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-xec",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "permitted"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            ).encode()
        )

    with wire_server(vendor) as policy, wire_server(provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "xecguard",
                    "mode": "post_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-xecguard-key",
                },
            }
        ]
        path: Final = tmp_path / "xecguard.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=upstream.url,
                api_key="synthetic-openai-key",
            )
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "permitted"
            scans: Final = tuple(request for request in policy.drain() if request.target == "/xecguard/v1/scan")
            assert scans, "post-call xecguard scan never reached the vendor"
            assert len(upstream.drain()) == 1
