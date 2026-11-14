import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_ASK: Final = {"role": "user", "content": "Design a Byzantine fault tolerant consensus protocol and prove it safe"}
_COMPACTION_PROMPT: Final = {"role": "user", "content": "Hello! Summarize the conversation so far."}


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


def _answering(content: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        return _completion(content)

    return respond


def _classifier(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    body: Final = request.body.decode()
    tier: Final = "SIMPLE" if "Summarize" in body else "REASONING"
    return _completion(json.dumps({"tier": tier}))


def test_codex_compaction_request_stays_on_the_held_user_turn_model(gateway: Gateway, tmp_path: Path) -> None:
    with (
        wire_server(_classifier) as judge,
        wire_server(_answering("simple answer")) as simple,
        wire_server(_answering("reasoning answer")) as reasoning,
    ):
        router: Final = "router-" + uuid.uuid4().hex[:8]
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": name,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": url, "api_key": "synthetic"},
            }
            for name, url in (("judge", judge.url), ("simple", simple.url), ("reasoning", reasoning.url))
        ] + [
            {
                "model_name": router,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "llm",
                        "classifier_llm_config": {"model": "judge", "timeout_ms": 20000},
                        "classification_mode": "user_turn",
                        "tiers": {
                            "SIMPLE": "simple",
                            "MEDIUM": "simple",
                            "COMPLEX": "reasoning",
                            "REASONING": "reasoning",
                        },
                    },
                },
            }
        ]
        path: Final = tmp_path / "complexity_router.yaml"
        path.write_text(yaml.safe_dump(config))
        session: Final = uuid.uuid4().hex

        def codex_headers(request_kind: str) -> dict[str, str]:
            return {
                "User-Agent": "codex-tui/0.164.0",
                "session_id": session,
                "x-codex-turn-metadata": json.dumps({"session_id": session, "request_kind": request_kind}),
            }

        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            ask: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": router, "messages": [_ASK]}, headers=codex_headers("turn")
            )
            assert ask.status_code == 200, ask.text
            assert ask.json()["choices"][0]["message"]["content"] == "reasoning answer", ask.text
            compaction: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": router, "messages": [_ASK, {"role": "assistant", "content": "done"}, _COMPACTION_PROMPT]},
                headers=codex_headers("compaction"),
            )
            assert compaction.status_code == 200, compaction.text
            assert compaction.json()["choices"][0]["message"]["content"] == "reasoning answer", compaction.text
            assert len([call for call in judge.drain() if call.method == "POST"]) == 1
            assert [call for call in simple.drain() if call.method == "POST"] == []
