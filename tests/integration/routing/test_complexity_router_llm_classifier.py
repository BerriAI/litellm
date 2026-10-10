import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


def _completion(content: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex[:8],
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            }
        ).encode()
    )


def _tier_model(answer: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        assert request.target == "/chat/completions", request.target
        return _completion(answer)

    return respond


def test_llm_classifier_verdict_routes_the_request_to_the_classified_tier_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    prompt: Final = f"hi there {uuid.uuid4().hex}"

    def classifier(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        assert request.target == "/chat/completions", request.target
        body: Final = json.loads(request.body)
        assert "response_format" in body, body
        assert [message["role"] for message in body["messages"]] == ["system", "user"], body
        assert prompt in json.dumps(body["messages"][1]), body
        return _completion(json.dumps({"tier": "COMPLEX"}))

    with (
        wire_server(classifier) as judge,
        wire_server(_tier_model("simple answer")) as simple,
        wire_server(_tier_model("complex answer")) as complex_tier,
    ):
        router: Final = "router-" + uuid.uuid4().hex[:8]
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": name,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": url, "api_key": "synthetic"},
            }
            for name, url in (("judge", judge.url), ("simple", simple.url), ("complex", complex_tier.url))
        ] + [
            {
                "model_name": router,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "llm",
                        "classifier_llm_config": {"model": "judge", "timeout_ms": 20000},
                        "tiers": {"SIMPLE": "simple", "MEDIUM": "simple", "COMPLEX": "complex", "REASONING": "complex"},
                    },
                },
            }
        ]
        path: Final = tmp_path / "complexity_router.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": router, "messages": [{"role": "user", "content": prompt}]}
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "complex answer", response.text
            assert len([call for call in judge.drain() if call.method == "POST"]) == 1
            assert [call for call in simple.drain() if call.method == "POST"] == []
            forwarded: Final = [call for call in complex_tier.drain() if call.method == "POST"]
            assert len(forwarded) == 1
            assert prompt in forwarded[0].body.decode()
