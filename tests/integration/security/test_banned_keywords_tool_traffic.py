from __future__ import annotations

import json
from collections.abc import Iterator
from functools import reduce
from typing import Final

import pytest
from integration.security.test_hide_secrets_tool_traffic import Rig, tool_rig
from pydantic import JsonValue

KEYWORD: Final = "zebracornflux"


def _ban_keyword(configuration: dict[str, JsonValue]) -> None:
    settings: Final = configuration["litellm_settings"]
    assert isinstance(settings, dict)
    settings["banned_keywords_list"] = [KEYWORD]
    settings["callbacks"] = ["banned_keywords"]
    del configuration["guardrails"]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with tool_rig(tmp_path_factory.mktemp("banned-keywords-tools"), configure=_ban_keyword) as value:
        yield value


def _chat(arguments: str, *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": "gpt",
        "stream": stream,
        "cache": {"no-cache": True},
        "messages": [
            {"role": "user", "content": "read .env"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": arguments}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
    }


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_keyword_in_tool_call_arguments_is_blocked(rig: Rig, stream: bool) -> None:
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "POST", "/v1/chat/completions", _chat(json.dumps({"path": f"notes/{KEYWORD}.txt"}), stream=stream)
    )
    assert response.status_code == 400, response.text
    assert f"Keyword={KEYWORD}" in response.text
    assert rig.wire.drain() == ()


def test_tool_call_arguments_without_keyword_reach_provider(rig: Rig) -> None:
    rig.wire.drain()
    arguments: Final = json.dumps({"path": "notes/plain.txt"})
    response: Final = rig.proxy.request("POST", "/v1/chat/completions", _chat(arguments))
    assert response.status_code == 200, response.text
    sent: Final = rig.wire.drain()
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
    assert rig.wire.drain() == ()
