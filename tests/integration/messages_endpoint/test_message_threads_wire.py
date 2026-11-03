import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

ANTHROPIC_MODEL: Final = "threads-anthropic"
BRIDGED_MODEL: Final = "threads-bridged"
MESSAGE_THREADS_BETA: Final = "message-threads-2026-08-12"
STALE_PREVIOUS_MESSAGE_ID: Final = "msg_01stale"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_NONCE: Final = re.compile(rb"nonce-([0-9a-f]{32})")


def nonce() -> str:
    return f"nonce-{uuid.uuid4().hex}"


def thread_not_found_reply() -> Reply:
    return Reply(
        status=404,
        body=json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": "No thread state was found for the requested `previous_message_id`.",
                    "details": {"error_code": "thread_not_found"},
                },
                "request_id": "req_upstream",
            }
        ).encode(),
    )


def respond(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    match: Final = _NONCE.search(request.body)
    assert match is not None, request.body[:200]
    identity: Final = match.group(1).decode()
    if STALE_PREVIOUS_MESSAGE_ID.encode() in request.body:
        return thread_not_found_reply()
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


def write_config(directory: Path, wire_url: str) -> Path:
    path: Final = directory / f"threads-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": ANTHROPIC_MODEL,
                        "litellm_params": {
                            "model": "anthropic/claude-opus-4-8",
                            "api_base": f"{wire_url}/anthropic",
                            "api_key": "synthetic-anthropic-key",
                        },
                    },
                    {
                        "model_name": BRIDGED_MODEL,
                        "litellm_params": {
                            "model": "openai/gpt-5.5",
                            "api_base": f"{wire_url}/openai",
                            "api_key": "synthetic-openai-key",
                        },
                    },
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

    def send(self, model: str, request_nonce: str, thread: Mapping[str, str], **fields: JsonValue) -> tuple[int, bytes]:
        response: Final = self.gateway.client.post(
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": f"hi {request_nonce}"}],
                "thread": dict(thread),
                **fields,
            },
            headers={"Authorization": f"Bearer {self.gateway.key}"},
            timeout=120,
        )
        return response.status_code, response.content

    def received(self, request_nonce: str) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if request_nonce.encode() in request.body)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("threads")
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


def beta_header(headers: Mapping[str, str]) -> frozenset[str]:
    return frozenset(
        piece for key, value in headers.items() if key.lower() == "anthropic-beta" for piece in value.split(",")
    )  # comprehension-ok: one header line may carry several comma-separated betas


def test_a_create_turn_reaches_anthropic_with_the_thread_field_and_its_beta(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(ANTHROPIC_MODEL, request_nonce, {"type": "create"}, system="be terse")
    assert status == 200, raw[:500]
    assert _JSON_OBJECT.validate_json(raw)["id"] == f"msg_{request_nonce.removeprefix('nonce-')}"
    (upstream,) = rig.received(request_nonce)
    body: Final = _JSON_OBJECT.validate_json(upstream.body)
    assert body["thread"] == {"type": "create"}
    assert body["system"] == "be terse"
    assert MESSAGE_THREADS_BETA in beta_header(upstream.headers), upstream.headers


def test_a_continue_turn_reaches_anthropic_with_only_the_new_messages_and_the_same_beta(rig: Rig) -> None:
    request_nonce: Final = nonce()
    thread: Final = {"type": "continue", "previous_message_id": "msg_01previous"}
    status, raw = rig.send(ANTHROPIC_MODEL, request_nonce, thread)
    assert status == 200, raw[:500]
    (upstream,) = rig.received(request_nonce)
    body: Final = _JSON_OBJECT.validate_json(upstream.body)
    assert body["thread"] == thread
    assert body["messages"] == [{"role": "user", "content": f"hi {request_nonce}"}]
    assert "system" not in body and "tools" not in body
    assert MESSAGE_THREADS_BETA in beta_header(upstream.headers), upstream.headers


def test_a_thread_on_a_translated_leg_is_a_400_and_never_reaches_the_provider(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(BRIDGED_MODEL, request_nonce, {"type": "continue", "previous_message_id": "msg_01p"})
    assert status == 400, raw[:500]
    error: Final = _JSON_OBJECT.validate_json(raw)
    assert "message threads are not supported" in json.dumps(error), error
    assert rig.received(request_nonce) == ()


def test_a_thread_not_found_from_anthropic_keeps_its_error_code_for_the_sdk_to_replay_on(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(
        ANTHROPIC_MODEL, request_nonce, {"type": "continue", "previous_message_id": STALE_PREVIOUS_MESSAGE_ID}
    )
    assert status == 404, raw[:500]
    error: Final = _JSON_OBJECT.validate_json(raw)
    detail: Final = error["error"]
    assert isinstance(detail, dict), error
    assert detail["type"] == "not_found_error"
    assert detail["details"] == {"error_code": "thread_not_found"}
