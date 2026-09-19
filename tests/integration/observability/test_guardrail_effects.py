import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.observability.guardrails.rewrite_reaches_correct_anthropic_positions")
def test_guardrail_rewrites_system_and_user_in_actual_anthropic_request(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    originals: Final = ["synthetic private system", "synthetic private user", "unchanged sibling"]
    replacements: Final = ["permitted system", "permitted user", "unchanged sibling"]

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        body: Final = json.loads(request.body)
        assert body["texts"] == originals
        return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "texts": replacements}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/messages"
        body: Final = json.loads(request.body)
        assert body["system"] == [{"type": "text", "text": replacements[0]}]
        assert body["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": replacements[1]}, {"type": "text", "text": replacements[2]}],
            }
        ]
        assert all(text.encode() not in request.body for text in originals[:2])
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "permitted response"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 4},
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        path: Final = tmp_path / "rewrite.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=upstream.url, api_key="synthetic-anthropic-key"
            )
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [
                        {"role": "system", "content": originals[0]},
                        {"role": "user", "content": [{"type": "text", "text": text} for text in originals[1:]]},
                    ],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "permitted response"
            assert response.json()["choices"][0]["finish_reason"] == "stop"
            assert response.json()["usage"]["total_tokens"] == 15
            assert len(policy.drain()) == len(upstream.drain()) == 1


@pytest.mark.covers("other.observability.guardrails.denial_prevents_provider_with_allowed_control")
def test_guardrail_denial_prevents_provider_and_preserves_allowed_control(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        body: Final = json.loads(request.body)
        assert body["texts"] in (["synthetic denied marker"], ["synthetic allowed marker"])
        result: Final = (
            {"action": "BLOCKED", "blocked_reason": "synthetic policy denial"}
            if body["texts"] == ["synthetic denied marker"]
            else {"action": "NONE"}
        )
        return Reply(body=json.dumps(result).encode())

    with wire_server(guardrail) as policy:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        path: Final = tmp_path / "deny.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            key: Final = scenario.key(models=[model])
            import httpx

            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                denied: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "synthetic denied marker"}]},
                    key=key,
                )
                assert denied.status_code == 400 and "synthetic policy denial" in denied.text, denied.text
                assert observed.get("/__observations").json()["requests"] == []
                allowed: Final = candidate.chat(model, text="synthetic allowed marker", key=key)
                assert allowed["usage"]["total_tokens"] == 40
                assert (
                    allowed["choices"][0]["message"]["content"]
                    == "Hello! This is a mock response from the fake OpenAI endpoint."
                )
                assert len(observed.get("/__observations").json()["requests"]) == 1
            assert len(policy.drain()) == 2
