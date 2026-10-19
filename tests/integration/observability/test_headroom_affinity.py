"""The Headroom guardrail tags its outbound calls with an opaque session affinity, through a real proxy.

`x-headroom-affinity` is derived from the validated virtual key and the client's own session id and
nothing else, so the compression service can pin related calls together without ever seeing either
raw value. The value must repeat for the same key and session, change when the key changes, ignore
identity fields a client forges into the request body, and be absent when the client sent no session
id. Everything is asserted on the requests the owned compression sink recorded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, object_value
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

AFFINITY_HEADER: Final = "x-headroom-affinity"
SESSION_HEADER: Final = "x-litellm-session-id"
COMPRESS_PATH: Final = "/v1/compress"
AFFINITY_SHAPE: Final = re.compile(r"[0-9a-f]{32}")
SESSION_ID: Final = "3f1c9b7e2d4a4c6e8b0f1a2c3d4e5f60"
FORGED_IDENTITY: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"user_api_key_team_id": "victim", "user_api_key_hash": "forged"}
)
MESSAGES: Final[tuple[dict[str, JsonValue], ...]] = (
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "Summarize what session affinity buys a compression service."},
    {"role": "assistant", "content": "It keeps related calls on one node so cached context is reused."},
    {"role": "user", "content": "And what identifies the session to that service?"},
)
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])


def _compress(request: Request) -> Reply:
    body: Final = _JSON.validate_json(request.body)
    return Reply(body=json.dumps({"messages": body["messages"], "tokens_before": 1000, "tokens_after": 100}).encode())


def _provider(_request: Request) -> Reply:
    return chat_reply("chatcmpl-headroom-affinity", "gpt-4o-mini", "ok", stream=False)


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    compress: Wire
    chat_model: str
    key_a: str
    key_b: str


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("headroom-affinity")
    with wire_server(_compress) as compress, wire_server(_provider) as provider:
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "headroom-affinity.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "litellm_settings": {**object_value(config["litellm_settings"]), "cache": False},
                    "guardrails": [
                        {
                            "guardrail_name": "headroom",
                            "litellm_params": {
                                "guardrail": "headroom",
                                "mode": "pre_call",
                                "default_on": True,
                                "api_base": compress.url,
                                "api_key": "synthetic-headroom-key",
                            },
                        }
                    ],
                }
            )
        )
        with (
            owned_proxy_process(session_gateway, directory, {}, config=path) as owned,
            owned.gateway.scenario() as scenario,
        ):
            chat_model: Final = scenario.model(api_base=provider.url + "/v1")
            yield Rig(owned, compress, chat_model, scenario.key(), scenario.key())


def _affinity_sent(
    rig: Rig, key: str, *, session: str | None, body: Mapping[str, JsonValue] = MappingProxyType({})
) -> str | None:
    rig.compress.drain()
    response: Final = rig.owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": list(MESSAGES), **body},
        key=key,
        headers={SESSION_HEADER: session} if session is not None else None,
    )
    assert response.status_code == 200, response.text
    compress_calls: Final = tuple(request for request in rig.compress.drain() if request.target == COMPRESS_PATH)
    assert len(compress_calls) == 1, [request.target for request in compress_calls]
    return compress_calls[0].headers.get(AFFINITY_HEADER)


def test_affinity_repeats_per_key_and_session_without_exposing_either(rig: Rig) -> None:
    first: Final = _affinity_sent(rig, rig.key_a, session=SESSION_ID)
    assert first is not None and AFFINITY_SHAPE.fullmatch(first), first
    assert _affinity_sent(rig, rig.key_a, session=SESSION_ID) == first
    assert first not in SESSION_ID
    assert first not in sha256(rig.key_a.encode()).hexdigest()
    assert _affinity_sent(rig, rig.key_a, session=SESSION_ID, body=FORGED_IDENTITY) == first
    other_key: Final = _affinity_sent(rig, rig.key_b, session=SESSION_ID)
    assert other_key is not None and AFFINITY_SHAPE.fullmatch(other_key), other_key
    assert other_key != first


def test_no_client_session_id_sends_no_affinity(rig: Rig) -> None:
    assert _affinity_sent(rig, rig.key_a, session=None) is None
