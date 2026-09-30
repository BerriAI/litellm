import json
import uuid
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_THROTTLED: Final = Reply(
    status=429,
    body=json.dumps({"message": "Rate exceeded"}).encode(),
    headers={"x-amzn-ErrorType": "ThrottlingException"},
)
_PASSED: Final = Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())
_PROVIDER: Final = Reply(
    body=json.dumps(
        {
            "id": "chatcmpl-bedrock-retry",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "retried"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    ).encode()
)


def _config(tmp_path: Path, guardrail_id: str, endpoint: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": "bedrock" + uuid.uuid4().hex,
            "litellm_params": {
                "guardrail": "bedrock",
                "mode": "pre_call",
                "default_on": True,
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "DRAFT",
                "aws_region_name": "us-east-1",
                "aws_access_key_id": "AKIASYNTHETICTHROTTLE",
                "aws_secret_access_key": "synthetic-secret",
                "aws_bedrock_runtime_endpoint": endpoint,
            },
        }
    ]
    path: Final = tmp_path / "bedrock-throttle.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_throttled_bedrock_guardrail_scan_is_retried_then_request_reaches_provider(
    gateway: Gateway, tmp_path: Path
) -> None:
    guardrail_id: Final = "throttle" + uuid.uuid4().hex[:8]
    prompt: Final = f"retry marker {uuid.uuid4().hex}"
    replies: Final = [_THROTTLED, _PASSED]

    def guardrail(request: Request) -> Reply:
        assert request.target == f"/guardrail/{guardrail_id}/version/DRAFT/apply", request.target
        assert json.loads(request.body)["source"] == "INPUT", request.body
        return replies.pop(0)

    with wire_server(guardrail) as policy, wire_server(lambda _request: _PROVIDER) as upstream:
        with (
            owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, guardrail_id, policy.url)) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=upstream.url, api_key="synthetic")
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}]}
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "retried"
            scans: Final = policy.drain()
            assert len(scans) == 2, scans
            assert scans[0].body == scans[1].body, "the retry must resend the identical scan"
            assert prompt in scans[1].body.decode()
            assert len(upstream.drain()) == 1


def test_bedrock_guardrail_throttle_that_never_clears_stops_after_the_retry_cap_without_calling_the_provider(
    gateway: Gateway, tmp_path: Path
) -> None:
    guardrail_id: Final = "throttle" + uuid.uuid4().hex[:8]

    with wire_server(lambda _request: _THROTTLED) as policy, wire_server(lambda _request: _PROVIDER) as upstream:
        with (
            owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, guardrail_id, policy.url)) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=upstream.url, api_key="synthetic")
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "capped"}]}
            )
            assert response.status_code == 429, response.text
            assert len(policy.drain()) == 4
            assert upstream.drain() == ()
