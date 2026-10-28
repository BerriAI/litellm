from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration._support.process import owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, wire_server

from tests.integration.cost_calculation.cost_tracking_case import SseResponse


def _frame(identity: str, content: str, finish: str | None = None) -> str:
    return "data: " + json.dumps(
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": finish,
                }
            ],
        }
    )


def _content(
    line: str,
    content_received: threading.Event,
    done_received: threading.Event,
) -> str:
    if not line.startswith("data:"):
        return ""
    payload: Final = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        done_received.set()
        return ""
    event: Final = JSON_OBJECT.validate_json(payload)
    assert "error" not in event, event
    pieces: Final = tuple(
        object_value(object_value(choice)["delta"]).get("content", "")
        for choice in list_value(event.get("choices", []))
    )
    assert all(isinstance(piece, str) for piece in pieces), event
    text: Final = "".join(piece for piece in pieces if isinstance(piece, str))
    if text:
        content_received.set()
    return text


def _read_stream(
    candidate: Gateway,
    model: str,
    content_received: threading.Event,
    done_received: threading.Event,
) -> str:
    with candidate.client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {candidate.key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "synthetic control"}],
            "stream": True,
        },
        timeout=30,
    ) as response:
        assert response.status_code == 200, response.read()
        return "".join(_content(line, content_received, done_received) for line in response.iter_lines())


@pytest.mark.parametrize("buffering", [None, True, False], ids=["omitted", "enabled", "disabled"])
def test_detect_only_decision_pipeline_respects_buffering(
    gateway: Gateway,
    tmp_path: Path,
    buffering: bool | None,
) -> None:
    identity: Final = f"decision-buffer-{uuid.uuid4().hex}"
    chat_model: Final = f"{identity}-chat"
    judge_model: Final = f"{identity}-judge"
    secret: Final = f"LIT9453_SECRET_{identity}"
    judge_entered: Final = threading.Event()
    release_judge: Final = threading.Event()
    content_received: Final = threading.Event()
    done_received: Final = threading.Event()

    def judge(request: Request) -> Reply:
        assert request.method == "POST", request
        assert request.target == "/v1/decisions", request
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert secret in json.dumps(body.get("input")), body
        judge_entered.set()
        assert release_judge.wait(timeout=20), "Judge gate was never released"
        return Reply(
            body=json.dumps(
                {
                    "model": "gpt-6-luna",
                    "answers": [
                        {
                            "type": "predicate",
                            "name": "unsafe_output",
                            "probability": 0.99,
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "total_tokens": 11,
                    },
                }
            ).encode()
        )

    with gateway.scenario() as scenario:
        upstream: Final = register_scenario(
            identity,
            SseResponse(
                content_type="text/event-stream",
                frames=(
                    _frame(identity, secret),
                    _frame(identity, "", "stop"),
                    "data: [DONE]",
                ),
                frame_delay_ms=20,
            ),
            control_url=gateway.upstream_url,
        )
        scenario.cleanups.callback(delete_scenario, upstream)

        with wire_server(judge) as judge_server:
            base: Final = JSON_OBJECT.validate_python(
                yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            )
            config: Final = {
                **base,
                "model_list": [
                    {
                        "model_name": chat_model,
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": f"{upstream.api_base()}/v1",
                            "api_key": "synthetic-chat-key",
                        },
                    },
                    {
                        "model_name": judge_model,
                        "litellm_params": {
                            "model": "openai/gpt-6-luna",
                            "api_base": f"{judge_server.url}/v1",
                            "api_key": "synthetic-judge-key",
                        },
                    },
                ],
                "guardrails": [
                    {
                        "guardrail_name": identity,
                        "litellm_params": {
                            "guardrail": "decision_model",
                            "mode": "post_call",
                            "default_on": False,
                            "decision_model": judge_model,
                            "timeout": 25,
                            "streaming_end_of_stream_only": True,
                            **({"streaming_buffer_until_moderated": buffering} if buffering is not None else {}),
                            "checks": [
                                {
                                    "name": "unsafe_output",
                                    "instructions": ("Does the response contain the synthetic secret?"),
                                    "action": "block",
                                    "threshold": 0.5,
                                }
                            ],
                        },
                    }
                ],
                "policies": {
                    identity: {
                        "guardrails": {"add": [identity]},
                        "pipeline": {
                            "mode": "post_call",
                            "steps": [
                                {
                                    "guardrail": identity,
                                    "on_pass": "allow",
                                    "on_fail": "next",
                                    "on_error": "next",
                                }
                            ],
                        },
                    }
                },
                "policy_attachments": [{"policy": identity, "models": [chat_model]}],
            }
            config_path: Final = tmp_path / "decision-pipeline-buffering.yaml"
            config_path.write_text(yaml.safe_dump(config))

            with (
                owned_proxy(
                    gateway,
                    tmp_path,
                    {},
                    config=config_path,
                    workers=2,
                ) as candidate,
                ThreadPoolExecutor(max_workers=1) as reader,
            ):
                result: Final = reader.submit(
                    _read_stream,
                    candidate,
                    chat_model,
                    content_received,
                    done_received,
                )
                try:
                    assert judge_entered.wait(timeout=15), "The completed chat response never reached the judge"
                    if buffering is False:
                        assert content_received.wait(timeout=5), (
                            "Pipeline did not release explicitly unbuffered content"
                        )
                    else:
                        assert not content_received.wait(timeout=1), (
                            "Pipeline released content while its judge was held"
                        )
                    assert not result.done(), "Client stream ended before the judge verdict"
                finally:
                    release_judge.set()

                assert result.result(timeout=15) == secret
                assert done_received.is_set(), "Missing terminal SSE [DONE]"
                requests: Final = judge_server.drain()
                assert len(requests) == 1, requests
                observed: Final = JSON_OBJECT.validate_json(requests[0].body)
                assert secret in json.dumps(observed.get("input")), observed
