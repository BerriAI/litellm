import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

API_KEY: Final = "wbk_eg_synthetic"
APP_ID: Final = "synthetic-app"
DENIED: Final = "synthetic wingback denied marker"
DENIAL: Final = "synthetic wingback denial"
PRE_CALL: Final = "wingback-pre"
POST_CALL: Final = "wingback-post"


def _verdict(request: Request) -> Reply:
    texts: Final = json.loads(request.body).get("texts") or []
    denied: Final = any(DENIED in text for text in texts)
    result: Final = {"action": "BLOCKED", "blocked_reason": DENIAL} if denied else {"action": "NONE"}
    return Reply(body=json.dumps(result).encode())


def _echo_provider(request: Request) -> Reply:
    prompt: Final = json.loads(request.body)["messages"][-1]["content"]
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": f"echo {prompt}"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _wingback(name: str, mode: str, policy: str) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "wingback",
            "mode": mode,
            "default_on": False,
            "api_base": policy,
            "api_key": API_KEY,
            "wingback_app_id": APP_ID,
        },
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    policy: Wire
    provider: Wire
    model: str


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("wingback")
    with (
        gateway_from_environment() as gateway,
        wire_server(_verdict) as policy,
        wire_server(_echo_provider) as provider,
    ):
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"]["cache"] = False
        config["guardrails"] = [
            _wingback(PRE_CALL, "pre_call", policy.url),
            _wingback(POST_CALL, "post_call", policy.url),
        ]
        path: Final = root / "wingback.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy_process(gateway, root, {}, config=path, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
            )
            yield Rig(owned.gateway, policy, provider, model)


def _chat(rig: Rig, guardrail: str, text: str) -> httpx.Response:
    rig.policy.drain()
    rig.provider.drain()
    return rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.model, "messages": [{"role": "user", "content": text}], "guardrails": [guardrail]},
    )


def _assert_wingback_request(request: Request, input_type: str, text: str) -> None:
    assert request.target == "/beta/litellm_basic_guardrail_api", request.target
    assert request.headers.get("x-api-key") == API_KEY
    body: Final = json.loads(request.body)
    assert body["input_type"] == input_type, body
    assert any(text in sent for sent in body["texts"]), body
    assert body["additional_provider_specific_params"] == {"wingback_app_id": APP_ID}, body


def test_pre_call_denial_never_reaches_the_provider(rig: Rig) -> None:
    response: Final = _chat(rig, PRE_CALL, f"{DENIED} {uuid.uuid4().hex}")

    assert response.status_code == 400 and DENIAL in response.text, response.text
    assert rig.provider.drain() == ()
    checks: Final = rig.policy.drain()
    assert len(checks) == 1, checks
    _assert_wingback_request(checks[0], "request", DENIED)


def test_post_call_denial_blocks_the_provider_response(rig: Rig) -> None:
    response: Final = _chat(rig, POST_CALL, f"{DENIED} {uuid.uuid4().hex}")

    assert response.status_code == 400 and DENIAL in response.text, response.text
    assert len(rig.provider.drain()) == 1
    checks: Final = rig.policy.drain()
    assert len(checks) == 1, checks
    _assert_wingback_request(checks[0], "response", f"echo {DENIED}")


def test_allowed_call_is_checked_once_and_logged_once(rig: Rig) -> None:
    prompt: Final = "synthetic wingback allowed prompt " + uuid.uuid4().hex
    response: Final = _chat(rig, PRE_CALL, prompt)

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == f"echo {prompt}"
    assert len(rig.provider.drain()) == 1
    assert response.headers["x-litellm-applied-guardrails"] == PRE_CALL, dict(response.headers)
    checks: Final = rig.policy.drain()
    assert len(checks) == 1, checks
    _assert_wingback_request(checks[0], "request", prompt)
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response.json()["id"],)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    entries: Final = object_value(rows[0]["metadata"])["guardrail_information"]
    assert isinstance(entries, list) and len(entries) == 1, entries
    entry: Final = object_value(entries[0])
    assert entry["guardrail_name"] == PRE_CALL, entry
    assert entry["guardrail_mode"] == "pre_call", entry
    assert entry["guardrail_status"] == "success", entry
