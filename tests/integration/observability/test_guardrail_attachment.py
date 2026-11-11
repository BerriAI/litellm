from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml

from tests.integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_ATTACHABLE: Final = "attachable-during-guard"
_WORDS: Final = "custom-words-during-guard"
_HEADER: Final = "x-litellm-applied-guardrails"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    policy: Wire
    upstream: Wire
    model: str


def _completion(request: Request) -> Reply:
    assert request.target == "/v1/chat/completions", request.target
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "guarded"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _allow(request: Request) -> Reply:
    assert request.target == "/beta/litellm_basic_guardrail_api", request.target
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _config(directory: Path, policy: Wire) -> Path:
    shutil.copy(Path("litellm/proxy/example_config_yaml/custom_guardrail.py"), directory / "custom_guardrail.py")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["guardrails"] = [
        {
            "guardrail_name": _ATTACHABLE,
            "litellm_params": {
                "guardrail": "generic_guardrail_api",
                "mode": "during_call",
                "api_base": policy.url,
                "api_key": "synthetic-guardrail-key",
            },
        },
        {
            "guardrail_name": _WORDS,
            "litellm_params": {"guardrail": "custom_guardrail.myCustomGuardrail", "mode": "during_call"},
        },
    ]
    path: Final = directory / "guardrail-attachment.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("guardrail-attachment")
    with (
        wire_server(_allow) as policy,
        wire_server(_completion) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=_config(directory, policy)) as owned,
        owned.scenario() as scenario,
    ):
        yield _Rig(owned, policy, upstream, scenario.model(api_base=f"{upstream.url}/v1", api_key="sk-fixture"))


def _prompt(text: str) -> str:
    return f"{text} {uuid.uuid4().hex}"


def _ask(rig: _Rig, key: str | None, prompt: str, guardrails: list[str] | None = None) -> httpx.Response:
    body: Final = {"model": rig.model, "messages": [{"role": "user", "content": prompt}]}
    return rig.gateway.request(
        "POST", "/v1/chat/completions", body if guardrails is None else {**body, "guardrails": guardrails}, key=key
    )


def _served_without_guardrail(rig: _Rig, response: httpx.Response, prompt: str) -> None:
    assert response.status_code == 200, response.text
    assert _HEADER not in response.headers, dict(response.headers)
    assert rig.policy.drain() == ()
    forwarded: Final = rig.upstream.drain()
    assert len(forwarded) == 1 and prompt in forwarded[0].body.decode(), forwarded


def _served_with_attachable(rig: _Rig, response: httpx.Response, prompt: str) -> None:
    assert response.status_code == 200, response.text
    assert response.headers[_HEADER] == _ATTACHABLE, dict(response.headers)
    inspected: Final = rig.policy.drain()
    assert len(inspected) == 1 and prompt in inspected[0].body.decode(), inspected
    forwarded: Final = rig.upstream.drain()
    assert len(forwarded) == 1 and prompt in forwarded[0].body.decode(), forwarded


def test_a_request_with_an_empty_guardrail_list_is_served_without_the_applied_header(rig: _Rig) -> None:
    prompt: Final = _prompt("no guardrails")
    _served_without_guardrail(rig, _ask(rig, None, prompt, []), prompt)


def test_a_key_carrying_a_guardrail_applies_it_and_a_plain_key_does_not(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        plain: Final = scenario.key()
        guarded: Final = scenario.key(guardrails=[_ATTACHABLE])
        plain_prompt: Final = _prompt("plain key")
        _served_without_guardrail(rig, _ask(rig, plain, plain_prompt), plain_prompt)
        guarded_prompt: Final = _prompt("guarded key")
        _served_with_attachable(rig, _ask(rig, guarded, guarded_prompt), guarded_prompt)


def test_a_team_carrying_a_guardrail_applies_it_to_its_keys_only(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(guardrails=[_ATTACHABLE])
        outside: Final = scenario.key()
        member: Final = scenario.key(team_id=team)
        outside_prompt: Final = _prompt("outside team")
        _served_without_guardrail(rig, _ask(rig, outside, outside_prompt), outside_prompt)
        member_prompt: Final = _prompt("team key")
        _served_with_attachable(rig, _ask(rig, member, member_prompt), member_prompt)


def test_a_during_call_custom_guardrail_rejects_a_request_naming_the_banned_word(rig: _Rig) -> None:
    unguarded_prompt: Final = _prompt("what is litellm")
    _served_without_guardrail(rig, _ask(rig, None, unguarded_prompt), unguarded_prompt)
    refused: Final = _ask(rig, None, _prompt("what is litellm"), [_WORDS])
    rig.upstream.drain()
    assert refused.status_code >= 400, refused.text
    error: Final = object_value(JSON_OBJECT.validate_json(refused.content)["error"])
    assert "Guardrail failed words - `litellm` detected" in string_value(error["message"]), refused.text
    assert rig.policy.drain() == ()
