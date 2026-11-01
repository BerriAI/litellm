import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment, list_value, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

ANTHROPIC_MODEL: Final = "beta-gated-anthropic"
FOUNDRY_MODEL: Final = "beta-gated-foundry"
FOUNDRY_DROP_PARAMS_MODEL: Final = "beta-gated-foundry-drop-params"
CLEAR_AT_BETA: Final = "mid-conversation-system-clear-at-2026-08-21"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_NONCE: Final = re.compile(rb"nonce-([0-9a-f]{32})")


def nonce() -> str:
    return f"nonce-{uuid.uuid4().hex}"


def claude_reply(request: Request) -> Reply:
    match: Final = _NONCE.search(request.body)
    assert match is not None, request.body[:200]
    identity: Final = match.group(1).decode()
    return Reply(
        body=json.dumps(
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": f"answered {identity}"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 3},
            }
        ).encode()
    )


def respond(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    return claude_reply(request)


def deployment(name: str, model: str, api_base: str, drop_params: bool = False) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": model,
            "api_base": api_base,
            "api_key": f"synthetic-{name}-key",
            **({"drop_params": True} if drop_params else {}),
        },
    }


def write_config(directory: Path, wire_url: str) -> Path:
    path: Final = directory / f"beta-gated-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            {
                "model_list": [
                    deployment(ANTHROPIC_MODEL, "anthropic/claude-opus-4-8", f"{wire_url}/anthropic"),
                    deployment(FOUNDRY_MODEL, "azure_ai/claude-opus-4-8", f"{wire_url}/foundry"),
                    deployment(
                        FOUNDRY_DROP_PARAMS_MODEL, "azure_ai/claude-opus-4-8", f"{wire_url}/foundry", drop_params=True
                    ),
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    wire: Wire

    def send(self, model: str, request_nonce: str) -> tuple[int, bytes]:
        response: Final = self.gateway.client.post(
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": f"hi {request_nonce}"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "how are you"},
                    {"role": "system", "content": "Answer in French", "clear_at": "next_user_message"},
                ],
            },
            headers={"Authorization": f"Bearer {self.gateway.key}", "anthropic-beta": CLEAR_AT_BETA},
            timeout=120,
        )
        return response.status_code, response.content

    def received(self, request_nonce: str) -> Request:
        requests: Final = tuple(request for request in self.wire.drain() if request_nonce.encode() in request.body)
        assert len(requests) == 1, [request.target for request in requests]
        return requests[0]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("beta-gated")
    with (
        gateway_from_environment() as shared,
        wire_server(respond) as wire,
        owned_proxy(
            shared,
            directory,
            {"LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "true"},
            config=write_config(directory, wire.url),
            workers=1,
        ) as gateway,
    ):
        yield Rig(gateway, wire)


def reply_text(status: int, raw: bytes) -> str:
    assert status == 200, raw[:500]
    return "".join(str(object_value(block)["text"]) for block in list_value(_JSON_OBJECT.validate_json(raw)["content"]))


def beta_header(headers: Mapping[str, str]) -> str:
    return ",".join(value for key, value in headers.items() if key.lower() == "anthropic-beta")


def system_message(body: bytes) -> dict[str, JsonValue]:
    messages: Final = list_value(_JSON_OBJECT.validate_json(body)["messages"])
    system: Final = [object_value(m) for m in messages if object_value(m).get("role") == "system"]
    assert len(system) == 1, messages
    return system[0]


def test_a_drop_params_leg_whose_row_lacks_the_beta_sends_neither_the_beta_nor_clear_at(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(FOUNDRY_DROP_PARAMS_MODEL, request_nonce)
    assert reply_text(status, raw) == f"answered {request_nonce.removeprefix('nonce-')}"
    upstream: Final = rig.received(request_nonce)
    assert CLEAR_AT_BETA not in beta_header(upstream.headers), upstream.headers
    assert system_message(upstream.body) == {"role": "system", "content": "Answer in French"}


def test_a_leg_whose_row_lacks_the_beta_forwards_clear_at_as_sent_without_drop_params(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(FOUNDRY_MODEL, request_nonce)
    assert reply_text(status, raw) == f"answered {request_nonce.removeprefix('nonce-')}"
    upstream: Final = rig.received(request_nonce)
    assert CLEAR_AT_BETA not in beta_header(upstream.headers), upstream.headers
    assert system_message(upstream.body) == {
        "role": "system",
        "content": "Answer in French",
        "clear_at": "next_user_message",
    }


def test_a_leg_that_carries_the_beta_forwards_clear_at_untouched(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(ANTHROPIC_MODEL, request_nonce)
    assert reply_text(status, raw) == f"answered {request_nonce.removeprefix('nonce-')}"
    upstream: Final = rig.received(request_nonce)
    assert CLEAR_AT_BETA in beta_header(upstream.headers), upstream.headers
    assert system_message(upstream.body) == {
        "role": "system",
        "content": "Answer in French",
        "clear_at": "next_user_message",
    }
