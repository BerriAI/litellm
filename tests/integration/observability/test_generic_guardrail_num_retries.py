"""generic_guardrail_api `num_retries` resends a check that timed out waiting for its answer, through a real proxy.

Each guardrail points at its own owned stub. A stub that holds a check never answers it before the 1s guardrail
timeout, and holding lasts until teardown so a resent check can only be answered by a later arrival.
"""

import itertools
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

MODEL: Final = "guarded-chat"
UPSTREAM_MODEL: Final = "openai/gpt-6.1-sol"
GUARDRAIL_TIMEOUT_SECONDS: Final = 1
HOLDS_FIRST_RETRIED: Final = "holds-first-check-retried-once"
HOLDS_FIRST_UNSET: Final = "holds-first-check-num-retries-unset"
HOLDS_ALL_RETRIED_TWICE: Final = "holds-every-check-retried-twice"
UNAVAILABLE_RETRIED_TWICE: Final = "unavailable-retried-twice"
ANSWER: Final = "upstream answer"
NO_ACTION: Final = Reply(body=json.dumps({"action": "NONE"}).encode())
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    upstream: Wire
    guardrails: Mapping[str, Wire]


def _holds_first(release: threading.Event) -> Callable[[Request], Reply]:
    arrivals: Final = itertools.count()

    def respond(_request: Request) -> Reply:
        if next(arrivals) == 0:
            release.wait()
        return NO_ACTION

    return respond


def _holds_all(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        release.wait()
        return NO_ACTION

    return respond


def _unavailable(_request: Request) -> Reply:
    return Reply(status=503, body=b'{"error": "guardrail unavailable"}')


def _upstream(_request: Request) -> Reply:
    return chat_reply("chatcmpl-num-retries", "gpt-6.1-sol", ANSWER, stream=False)


def _guardrail(name: str, wire: Wire, num_retries: int | None) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": False,
            "api_base": wire.url,
            "api_key": "synthetic-guardrail-key",
            "timeout": GUARDRAIL_TIMEOUT_SECONDS,
            **({} if num_retries is None else {"num_retries": num_retries}),
        },
    }


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("generic-guardrail-num-retries")
    release: Final = threading.Event()
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(wire_server(_upstream))
        guardrails: Final = MappingProxyType(
            {
                HOLDS_FIRST_RETRIED: stack.enter_context(wire_server(_holds_first(release), keep_alive=True)),
                HOLDS_FIRST_UNSET: stack.enter_context(wire_server(_holds_first(release), keep_alive=True)),
                HOLDS_ALL_RETRIED_TWICE: stack.enter_context(wire_server(_holds_all(release), keep_alive=True)),
                UNAVAILABLE_RETRIED_TWICE: stack.enter_context(wire_server(_unavailable, keep_alive=True)),
            }
        )
        stack.callback(release.set)
        retries: Final = MappingProxyType(
            {HOLDS_FIRST_RETRIED: 1, HOLDS_FIRST_UNSET: None, HOLDS_ALL_RETRIED_TWICE: 2, UNAVAILABLE_RETRIED_TWICE: 2}
        )
        base: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "generic-guardrail-num-retries.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **base,
                    "model_list": [
                        {
                            "model_name": MODEL,
                            "litellm_params": {
                                "model": UPSTREAM_MODEL,
                                "api_base": f"{upstream.url}/v1",
                                "api_key": "synthetic-openai-key",
                            },
                        }
                    ],
                    "guardrails": [_guardrail(name, wire, retries[name]) for name, wire in guardrails.items()],
                }
            )
        )
        owned: Final = stack.enter_context(owned_proxy_process(session_gateway, directory, {}, config=path))
        yield Rig(owned, upstream, guardrails)


def _chat(rig: Rig, guardrail: str) -> httpx.Response:
    return rig.owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "is this allowed?"}], "guardrails": [guardrail]},
    )


def _checks(wire: Wire) -> tuple[bytes, ...]:
    return tuple(request.body for request in wire.drain())


def test_check_held_past_the_timeout_is_resent_on_a_fresh_connection_and_the_request_is_served(rig: Rig) -> None:
    response: Final = _chat(rig, HOLDS_FIRST_RETRIED)

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == ANSWER
    stub: Final = rig.guardrails[HOLDS_FIRST_RETRIED]
    checks: Final = _checks(stub)
    assert len(checks) == 2, checks
    assert checks[0] == checks[1]
    assert stub.connections() == 2
    assert len(rig.upstream.drain()) == 1


def test_check_held_past_the_timeout_is_sent_once_when_num_retries_is_unset(rig: Rig) -> None:
    response: Final = _chat(rig, HOLDS_FIRST_UNSET)

    assert response.status_code != 200, response.text
    assert "Generic Guardrail API failed" in response.text
    assert len(_checks(rig.guardrails[HOLDS_FIRST_UNSET])) == 1
    assert rig.upstream.drain() == ()


def test_num_retries_bounds_how_many_times_a_held_check_is_sent(rig: Rig) -> None:
    response: Final = _chat(rig, HOLDS_ALL_RETRIED_TWICE)

    assert response.status_code != 200, response.text
    assert "Generic Guardrail API failed" in response.text
    assert len(_checks(rig.guardrails[HOLDS_ALL_RETRIED_TWICE])) == 3
    assert rig.upstream.drain() == ()


def test_error_status_from_the_guardrail_is_not_resent(rig: Rig) -> None:
    response: Final = _chat(rig, UNAVAILABLE_RETRIED_TWICE)

    assert response.status_code != 200, response.text
    assert "Generic Guardrail API failed" in response.text
    assert len(_checks(rig.guardrails[UNAVAILABLE_RETRIED_TWICE])) == 1
    assert rig.upstream.drain() == ()
