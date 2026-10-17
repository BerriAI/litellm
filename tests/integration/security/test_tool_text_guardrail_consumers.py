from __future__ import annotations

import json
from collections.abc import Iterator
from functools import reduce
from typing import Final

import pytest
from integration._support.wire import Reply, Request
from integration.security.test_hide_secrets_tool_traffic import Rig, _provider, tool_rig
from pydantic import JsonValue

KEYWORD: Final = "zebracornflux"
FLAGGED: Final = "moderation-flagged-fragment"


def _vendors(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    body: Final = json.loads(request.body)
    if request.target.endswith("/moderations"):
        flagged: Final = FLAGGED in body["input"]
        return Reply(
            body=json.dumps(
                {
                    "id": "modr-integration",
                    "model": body["model"],
                    "results": [
                        {
                            "flagged": flagged,
                            "categories": {"violence": flagged},
                            "category_scores": {"violence": 0.99 if flagged else 0.0},
                        }
                    ],
                }
            ).encode()
        )
    if "/api/v1/text/contents" in request.target:
        return Reply(body=json.dumps([[] for _ in body["contents"]]).encode())
    if request.target.endswith("/api/v2/text/detection/content"):
        return Reply(body=b'{"detections": []}')
    return _provider(request)


def _consumers(configuration: dict[str, JsonValue]) -> None:
    models: Final = configuration["model_list"]
    settings: Final = configuration["litellm_settings"]
    assert isinstance(models, list) and isinstance(settings, dict)
    vendor: Final = next(model for model in models if isinstance(model, dict) and model["model_name"] == "gpt")
    assert isinstance(vendor, dict) and isinstance(vendor["litellm_params"], dict)
    base_url: Final = vendor["litellm_params"]["api_base"]
    models.append(
        {
            "model_name": "omni-moderation-latest",
            "litellm_params": {
                "model": "openai/omni-moderation-latest",
                "api_key": "synthetic-provider-key",
                "api_base": base_url,
            },
        }
    )
    settings["banned_keywords_list"] = [KEYWORD]
    settings["callbacks"] = ["banned_keywords", "openai_moderations"]
    configuration["guardrails"] = [
        {
            "guardrail_name": f"ibm-{name}",
            "litellm_params": {
                "guardrail": "ibm_guardrails",
                "mode": "pre_call",
                "default_on": True,
                "auth_token": "synthetic-detector-token",
                "base_url": base_url,
                "detector_id": "tool-text",
                "is_detector_server": detector_server,
            },
        }
        for name, detector_server in (("detector", True), ("orchestrator", False))
    ]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with tool_rig(tmp_path_factory.mktemp("tool-text-consumers"), _vendors, configure=_consumers) as value:
        yield value


def _chat(arguments: str, *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": "gpt",
        "stream": stream,
        "cache": {"no-cache": True},
        "messages": [
            {"role": "user", "content": "write notes"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "written"},
            {"role": "user", "content": "say hi"},
        ],
    }


def _messages(result: str) -> dict[str, JsonValue]:
    return {
        "model": "claude",
        "max_tokens": 32,
        "cache": {"no-cache": True},
        "system": [{"type": "text", "text": "system note"}],
        "messages": [
            {"role": "user", "content": "read notes"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "read_file",
                        "input": {"path": "notes.txt", "options": [7, {"mode": "tool input"}]},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": [{"type": "text", "text": result}]},
                    {"type": "text", "text": "summarize"},
                ],
            },
        ],
    }


def _provider_calls(sent: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in sent if request.target.endswith(("/chat/completions", "/messages")))


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_keyword_in_tool_call_arguments_is_blocked(rig: Rig, stream: bool) -> None:
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "POST", "/v1/chat/completions", _chat(json.dumps({"path": f"notes/{KEYWORD}.txt"}), stream=stream)
    )
    assert response.status_code == 400, response.text
    assert f"Keyword={KEYWORD}" in response.text
    assert _provider_calls(rig.wire.drain()) == ()


def test_tool_call_arguments_without_keyword_reach_provider(rig: Rig) -> None:
    rig.wire.drain()
    arguments: Final = json.dumps({"path": "notes/plain.txt"})
    response: Final = rig.proxy.request("POST", "/v1/chat/completions", _chat(arguments))
    assert response.status_code == 200, response.text
    sent: Final = _provider_calls(rig.wire.drain())
    assert len(sent) == 1
    assert json.loads(sent[0].body)["messages"][1]["tool_calls"][0]["function"]["arguments"] == arguments


def test_deep_tool_result_chain_cannot_hide_a_later_keyword(rig: Rig) -> None:
    rig.wire.drain()
    chain: Final = reduce(
        lambda content, _: [{"type": "tool_result", "tool_use_id": "tool_1", "content": content}],
        range(1000),
        [{"type": "text", "text": "inner"}],
    )
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt",
            "cache": {"no-cache": True},
            "messages": [{"role": "assistant", "content": chain}, {"role": "user", "content": f"say {KEYWORD}"}],
        },
    )
    assert response.status_code == 400, response.text
    assert f"Keyword={KEYWORD}" in response.text
    assert _provider_calls(rig.wire.drain()) == ()


@pytest.mark.parametrize(
    ("path", "body", "fragments"),
    (
        pytest.param(
            "/v1/messages",
            _messages("tool result"),
            ("system note", "read notes", "notes.txt\ntool input", "tool result", "summarize"),
            id="messages",
        ),
        pytest.param(
            "/v1/chat/completions",
            _chat(json.dumps({"path": "n.txt", "content": "argument text"})),
            ("write notes", "n.txt\nargument text", "written", "say hi"),
            id="chat",
        ),
    ),
)
def test_read_only_vendors_receive_tool_and_system_text(
    rig: Rig, path: str, body: dict[str, JsonValue], fragments: tuple[str, ...]
) -> None:
    rig.wire.drain()
    response: Final = rig.proxy.request("POST", path, body)
    assert response.status_code == 200, response.text
    sent: Final = rig.wire.drain()
    moderated: Final = tuple(json.loads(r.body)["input"] for r in sent if r.target.endswith("/moderations"))
    detected: Final = tuple(json.loads(r.body)["contents"] for r in sent if "/api/v1/text/contents" in r.target)
    orchestrated: Final = tuple(json.loads(r.body)["content"] for r in sent if r.target.endswith("/detection/content"))
    assert moderated == ("".join(fragments),)
    assert detected == (list(fragments),)
    assert orchestrated == fragments
    assert len(_provider_calls(sent)) == 1


def test_moderation_flag_in_tool_result_blocks(rig: Rig) -> None:
    response: Final = rig.proxy.request("POST", "/v1/messages", _messages(FLAGGED))
    assert response.status_code == 403, response.text
    assert "Violated content safety policy" in response.text
