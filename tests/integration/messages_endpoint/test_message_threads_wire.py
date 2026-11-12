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
BRIDGED_DROP_PARAMS_MODEL: Final = "threads-bridged-drop-params"
PASSTHROUGH_MODEL: Final = "threads-messages-passthrough"
MESSAGE_THREADS_BETA: Final = "message-threads-2026-08-12"
EXPIRED_THREAD_MARKER: Final = "thread-expired"
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


def anthropic_message(identity: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": f"answered {identity}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def sse_event(event: str, data: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def streamed_anthropic_reply(identity: str) -> Reply:
    message_start: Final = sse_event(
        "message_start", {"type": "message_start", "message": {**anthropic_message(identity), "content": []}}
    )
    return Reply(
        chunks=(
            message_start[:40],
            message_start[40:],
            sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            ),
            sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            ),
            sse_event("message_stop", {"type": "message_stop"}),
        ),
        content_type="text/event-stream",
    )


def responses_reply(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1789788253,
                "status": "completed",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{identity}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"answered {identity}", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
            }
        ).encode()
    )


def respond(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    match: Final = _NONCE.search(request.body)
    assert match is not None, request.body[:200]
    identity: Final = match.group(1).decode()
    if request.target.startswith("/openai"):
        return responses_reply(identity)
    if EXPIRED_THREAD_MARKER.encode() in request.body:
        return thread_not_found_reply()
    if b'"stream": true' in request.body or b'"stream":true' in request.body:
        return streamed_anthropic_reply(identity)
    return Reply(body=json.dumps(anthropic_message(identity)).encode())


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
                    {
                        "model_name": BRIDGED_DROP_PARAMS_MODEL,
                        "litellm_params": {
                            "model": "openai/gpt-5.5",
                            "api_base": f"{wire_url}/openai",
                            "api_key": "synthetic-openai-key",
                            "drop_params": True,
                        },
                    },
                    {
                        "model_name": PASSTHROUGH_MODEL,
                        "litellm_params": {
                            "model": "hosted_vllm/qwen3-coder",
                            "api_base": f"{wire_url}/vllm",
                            "api_key": "synthetic-vllm-key",
                        },
                        "model_info": {"supported_endpoints": ["/v1/messages"]},
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

    def send(
        self,
        model: str,
        request_nonce: str,
        thread: Mapping[str, str],
        *,
        key: str | None = None,
        text: str = "hi",
        **fields: JsonValue,
    ) -> tuple[int, bytes]:
        response: Final = self.gateway.client.post(
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": f"{text} {request_nonce}"}],
                "thread": dict(thread),
                **fields,
            },
            headers={"Authorization": f"Bearer {self.gateway.key if key is None else key}"},
            timeout=120,
        )
        return response.status_code, response.content

    def create(self, *, key: str | None = None, **fields: JsonValue) -> str:
        request_nonce: Final = nonce()
        status, raw = self.send(ANTHROPIC_MODEL, request_nonce, {"type": "create"}, key=key, **fields)
        assert status == 200, raw[:500]
        self.received(request_nonce)
        return f"msg_{request_nonce.removeprefix('nonce-')}"

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
            {
                "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "true",
                "ANTHROPIC_API_BASE": f"{wire.url}/anthropic",
                "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
            },
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
    thread: Final = {"type": "continue", "previous_message_id": rig.create()}
    status, raw = rig.send(ANTHROPIC_MODEL, request_nonce, thread)
    assert status == 200, raw[:500]
    (upstream,) = rig.received(request_nonce)
    body: Final = _JSON_OBJECT.validate_json(upstream.body)
    assert body["thread"] == thread
    assert body["messages"] == [{"role": "user", "content": f"hi {request_nonce}"}]
    assert "system" not in body and "tools" not in body
    assert MESSAGE_THREADS_BETA in beta_header(upstream.headers), upstream.headers


def assert_thread_not_found(status: int, raw: bytes) -> None:
    assert status == 404, raw[:500]
    error: Final = _JSON_OBJECT.validate_json(raw)
    detail: Final = error["error"]
    assert isinstance(detail, dict), error
    assert detail["type"] == "not_found_error"
    assert detail["details"] == {"error_code": "thread_not_found"}


@pytest.mark.parametrize("model", [BRIDGED_MODEL, PASSTHROUGH_MODEL], ids=["translated", "messages-passthrough"])
def test_a_create_turn_on_a_leg_without_threads_is_thread_unsupported_and_never_reaches_the_provider(
    rig: Rig, model: str
) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(model, request_nonce, {"type": "create"})
    assert status == 400, raw[:500]
    detail: Final = _JSON_OBJECT.validate_json(raw)["error"]
    assert isinstance(detail, dict), raw[:500]
    assert detail["type"] == "invalid_request_error"
    assert detail["details"] == {"error_code": "thread_unsupported_request"}
    assert rig.received(request_nonce) == ()


def test_a_create_turn_on_a_translated_leg_is_served_without_the_thread_under_drop_params(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(BRIDGED_DROP_PARAMS_MODEL, request_nonce, {"type": "create"})
    assert status == 200, raw[:500]
    (upstream,) = rig.received(request_nonce)
    assert upstream.target.startswith("/openai"), upstream.target
    assert "thread" not in _JSON_OBJECT.validate_json(upstream.body)


@pytest.mark.parametrize("model", [BRIDGED_MODEL, PASSTHROUGH_MODEL], ids=["translated", "messages-passthrough"])
def test_a_continue_turn_on_a_leg_without_threads_is_thread_not_found_and_never_reaches_the_provider(
    rig: Rig, model: str
) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(model, request_nonce, {"type": "continue", "previous_message_id": rig.create()})
    assert_thread_not_found(status, raw)
    assert rig.received(request_nonce) == ()


def test_another_key_cannot_continue_a_thread_and_never_reaches_anthropic(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        owner: Final = scenario.key()
        intruder: Final = scenario.key()
        thread: Final = {"type": "continue", "previous_message_id": rig.create(key=owner)}
        intruder_nonce: Final = nonce()
        assert_thread_not_found(*rig.send(ANTHROPIC_MODEL, intruder_nonce, thread, key=intruder))
        assert rig.received(intruder_nonce) == ()
        owner_nonce: Final = nonce()
        status, raw = rig.send(ANTHROPIC_MODEL, owner_nonce, thread, key=owner)
        assert status == 200, raw[:500]
        assert len(rig.received(owner_nonce)) == 1


def test_another_key_cannot_continue_a_thread_through_chat_completions(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        owner: Final = scenario.key()
        intruder: Final = scenario.key()
        thread: Final = {"type": "continue", "previous_message_id": rig.create(key=owner)}
        intruder_nonce: Final = nonce()
        response: Final = rig.gateway.client.post(
            "/v1/chat/completions",
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": f"hi {intruder_nonce}"}],
                "thread": thread,
            },
            headers={"Authorization": f"Bearer {intruder}"},
            timeout=120,
        )
        assert response.status_code == 404, response.content[:500]
        assert rig.received(intruder_nonce) == ()


ADVISOR_TOOL: Final = {"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-8"}
ADVISOR_BETA: Final = "advisor-tool-2026-03-01"


@pytest.mark.parametrize("stream", [False, True], ids=["json", "streamed"])
def test_a_continue_turn_resends_the_betas_its_create_turn_took_from_its_tools(rig: Rig, stream: bool) -> None:
    create_nonce: Final = nonce()
    status, raw = rig.send(ANTHROPIC_MODEL, create_nonce, {"type": "create"}, tools=[ADVISOR_TOOL], stream=stream)
    assert status == 200, raw[:500]
    (created,) = rig.received(create_nonce)
    create_betas: Final = beta_header(created.headers)
    assert {ADVISOR_BETA, MESSAGE_THREADS_BETA} <= create_betas, created.headers
    continue_nonce: Final = nonce()
    thread: Final = {"type": "continue", "previous_message_id": f"msg_{create_nonce.removeprefix('nonce-')}"}
    status, raw = rig.send(ANTHROPIC_MODEL, continue_nonce, thread)
    assert status == 200, raw[:500]
    (continued,) = rig.received(continue_nonce)
    assert "tools" not in _JSON_OBJECT.validate_json(continued.body)
    assert beta_header(continued.headers) == create_betas, continued.headers


def pass_through_continue(rig: Rig, request_nonce: str, previous_message_id: str, key: str) -> int:
    return rig.gateway.client.post(
        "/anthropic/v1/messages",
        json={
            "model": "claude-opus-4-8",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": f"hi {request_nonce}"}],
            "thread": {"type": "continue", "previous_message_id": previous_message_id},
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    ).status_code


def test_another_key_cannot_continue_a_thread_through_the_anthropic_pass_through_route(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        owner: Final = scenario.key()
        intruder: Final = scenario.key()
        previous_message_id: Final = rig.create(key=owner)
        intruder_nonce: Final = nonce()
        assert pass_through_continue(rig, intruder_nonce, previous_message_id, intruder) == 404
        assert rig.received(intruder_nonce) == ()
        owner_nonce: Final = nonce()
        assert pass_through_continue(rig, owner_nonce, previous_message_id, owner) == 200
        assert len(rig.received(owner_nonce)) == 1


def test_a_streamed_create_turn_can_be_continued_by_its_key(rig: Rig) -> None:
    previous_message_id: Final = rig.create(stream=True)
    request_nonce: Final = nonce()
    status, raw = rig.send(
        ANTHROPIC_MODEL, request_nonce, {"type": "continue", "previous_message_id": previous_message_id}
    )
    assert status == 200, raw[:500]
    assert len(rig.received(request_nonce)) == 1


def test_a_thread_not_found_from_anthropic_keeps_its_error_code_for_the_sdk_to_replay_on(rig: Rig) -> None:
    request_nonce: Final = nonce()
    thread: Final = {"type": "continue", "previous_message_id": rig.create()}
    assert_thread_not_found(*rig.send(ANTHROPIC_MODEL, request_nonce, thread, text=EXPIRED_THREAD_MARKER))
    assert len(rig.received(request_nonce)) == 1
