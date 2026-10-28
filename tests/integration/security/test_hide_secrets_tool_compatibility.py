from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.budget import case_timeout_seconds
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.security.test_hide_secrets_tool_traffic import (
    SECRET,
    TEXT,
    Rig,
    _input,
    _messages,
    _provider,
    tool_rig,
)
from pydantic import JsonValue

_MARKER: Final = re.compile(r"case_[0-9a-f]{32}")


def _identity() -> str:
    return f"case_{uuid.uuid4().hex}"


def _peer(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    body: Final = json.loads(request.body)
    marker: Final = _MARKER.search(request.body.decode())
    assert marker is not None, request.body
    identity: Final = marker.group()
    if request.target.endswith("/messages") and (
        not isinstance(body.get("messages"), list)
        or any(not isinstance(message, dict) for message in body.get("messages", []))
        or ("system" in body and not isinstance(body["system"], (str, list)))
    ):
        return Reply(
            status=400,
            body=b'{"type":"error","error":{"type":"invalid_request_error","message":"invalid shape"}}',
        )
    if "provider-error" in request.body.decode():
        return Reply(
            status=401,
            body=b'{"type":"error","error":{"message":"scripted deployment refused","type":"authentication_error"}}',
        )
    if request.target.endswith("/responses"):
        answer: Final = {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": body["model"],
            "error": None,
            "incomplete_details": None,
            "output": [
                {
                    "id": f"msg_{identity}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "accepted", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
        }
        return Reply(body=json.dumps(answer).encode())
    if body.get("stream") and not request.target.endswith("/messages"):
        chunk: Final = {
            "id": f"chatcmpl_{identity}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "accepted"}, "finish_reason": None}],
        }
        finish: Final = {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
        return Reply(
            content_type="text/event-stream",
            chunks=(
                f"data: {json.dumps(chunk)}\n\n".encode(),
                f"data: {json.dumps(finish)}\n\n".encode(),
                b"data: [DONE]\n\n",
            ),
        )
    reply: Final = _provider(request)
    if reply.chunks is not None:
        return replace(
            reply, chunks=tuple(re.sub(rb"msg_[a-z0-9_]+", f"msg_{identity}".encode(), chunk) for chunk in reply.chunks)
        )
    answer_body: Final = json.loads(reply.body)
    return replace(
        reply,
        body=json.dumps(
            {**answer_body, "id": f"{'msg' if request.target.endswith('/messages') else 'chatcmpl'}_{identity}"}
        ).encode(),
    )


@dataclass(frozen=True, slots=True)
class Caller:
    rig: Rig
    key: str


@contextmanager
def _key(rig: Rig, *, permissions: dict[str, JsonValue] | None = None) -> Iterator[str]:
    created: Final = rig.proxy.post(
        "/key/generate", {"models": ["*"], **({"permissions": permissions} if permissions else {})}
    )
    key: Final = created["key"]
    assert isinstance(key, str), created
    try:
        yield key
    finally:
        rig.proxy.post("/key/delete", {"keys": [key]})
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s',
                (sha256(key.encode()).hexdigest(),),
                database_url=rig.database_url,
            )
            == []
        )


@pytest.fixture(scope="module")
def caller(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Caller]:
    with tool_rig(tmp_path_factory.mktemp("hide-secrets-compat"), _peer) as rig:
        with _key(rig) as key:
            yield Caller(rig, key)


@pytest.fixture
def isolated_caller(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Caller]:
    with tool_rig(tmp_path_factory.mktemp("hide-secrets-worker-recovery"), _peer) as rig:
        with _key(rig) as key:
            yield Caller(rig, key)


def _body(provider: str, marker: str, *, stream: bool = False, secret: bool = True) -> dict[str, JsonValue]:
    if provider == "anthropic":
        body: Final = _messages("result-nested")
        body["messages"][0]["content"] = f"read .env {marker}"
        return {**body, "stream": stream, "cache": {"no-cache": True}}
    return {
        "model": "gpt",
        "stream": stream,
        "cache": {"no-cache": True},
        "messages": [
            {"role": "user", "content": f"read .env {marker}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(_input() if secret else {"path": ".env"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
    }


def _wire(caller: Caller, marker: str) -> dict[str, JsonValue]:
    requests: Final = caller.rig.wire.drain()
    selected: Final = tuple(request for request in requests if marker.encode() in request.body)
    assert len(selected) == 1, [(request.target, request.body) for request in requests]
    assert SECRET.encode() not in selected[0].body, selected[0].body
    assert b"REDACTED" in selected[0].body, selected[0].body
    return json.loads(selected[0].body)


def _stored(caller: Caller, response_id: str, *, masked: bool = True, cache: bool = False) -> dict[str, JsonValue]:
    query: Final = (
        'SELECT request_id, status, metadata, cache_hit FROM "LiteLLM_SpendLogs" '
        "WHERE request_id=%s OR litellm_call_id=%s"
        if not cache
        else 'SELECT request_id, status, metadata, cache_hit FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s'
    )
    rows: Final = eventually(
        lambda: read_rows(
            query,
            (response_id, response_id) if not cache else (f"{response_id}_cache_hit%",),
            database_url=caller.rig.database_url,
        ),
        lambda observed: len(observed) == 1,
        seconds=70,
    )
    information: Final = rows[0]["metadata"].get("guardrail_information") or []
    matching: Final = [item for item in information if item.get("guardrail_name") == "hide-secrets"]
    if masked:
        assert len(matching) == 1, rows[0]
        assert matching[0]["guardrail_status"] == "success", rows[0]
        assert sum((matching[0].get("masked_entity_count") or {}).values()) > 0, rows[0]
    else:
        assert not any(item.get("masked_entity_count") for item in matching), rows[0]
    if cache:
        assert str(rows[0]["cache_hit"]).lower() == "true", rows[0]
    return rows[0]


@pytest.mark.parametrize("provider", ("anthropic", "openai"))
@pytest.mark.parametrize("client_mode", ("sync", "async"))
@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.asyncio
async def test_sdk_tool_traffic_is_redacted(caller: Caller, provider: str, client_mode: str, stream: bool) -> None:
    marker: Final = _identity()
    body: Final = _body(provider, marker, stream=stream)
    options: Final = {"extra_headers": {"x-litellm-call-id": marker}, "extra_body": {"cache": body["cache"]}}
    if provider == "anthropic":
        arguments: Final = {
            "model": "claude",
            "max_tokens": 32,
            "messages": body["messages"],
            "stream": stream,
            **options,
        }
        if client_mode == "sync":
            with anthropic.Anthropic(
                api_key=caller.key,
                base_url=str(caller.rig.proxy.client.base_url),
                max_retries=0,
                http_client=httpx.Client(trust_env=False),
            ) as sdk:
                response = sdk.messages.create(**arguments)
                text = "".join(event.model_dump_json() for event in response) if stream else response.model_dump_json()
        else:
            async with anthropic.AsyncAnthropic(
                api_key=caller.key,
                base_url=str(caller.rig.proxy.client.base_url),
                max_retries=0,
                http_client=httpx.AsyncClient(trust_env=False),
            ) as sdk:
                response = await sdk.messages.create(**arguments)
                text = (
                    "".join([event.model_dump_json() async for event in response])
                    if stream
                    else response.model_dump_json()
                )
        response_id = f"msg_{marker}"
    else:
        arguments = {"model": "gpt", "messages": body["messages"], "stream": stream, **options}
        if client_mode == "sync":
            with openai.OpenAI(
                api_key=caller.key,
                base_url=str(caller.rig.proxy.client.base_url).rstrip("/") + "/v1",
                max_retries=0,
                http_client=httpx.Client(trust_env=False),
            ) as sdk:
                response = sdk.chat.completions.create(**arguments)
                text = "".join(event.model_dump_json() for event in response) if stream else response.model_dump_json()
        else:
            async with openai.AsyncOpenAI(
                api_key=caller.key,
                base_url=str(caller.rig.proxy.client.base_url).rstrip("/") + "/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(trust_env=False),
            ) as sdk:
                response = await sdk.chat.completions.create(**arguments)
                text = (
                    "".join([event.model_dump_json() async for event in response])
                    if stream
                    else response.model_dump_json()
                )
        response_id = f"chatcmpl_{marker}"
    assert "accepted" in text and response_id in text, text
    _wire(caller, marker)
    _stored(caller, response_id)


@pytest.mark.parametrize("path", ("/v1/messages", "/anthropic/v1/messages"), ids=("messages", "anthropic-alias"))
@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_messages_aliases_redact_system_and_multiple_tool_items(caller: Caller, path: str, stream: bool) -> None:
    marker: Final = _identity()
    body: Final = _body("anthropic", marker, stream=stream)
    if path.startswith("/anthropic/"):
        body["model"] = "claude-haiku-5-5"
    body["system"] = [{"type": "text", "text": TEXT, "cache_control": {"type": "ephemeral"}}]
    body["messages"][-1]["content"][0]["content"].append({"type": "text", "text": TEXT})
    response: Final = caller.rig.proxy.request(
        "POST", path, body, key=caller.key, headers={"x-litellm-call-id": marker}
    )
    assert response.status_code == 200 and "accepted" in response.text, response.text
    observed: Final = _wire(caller, marker)
    assert observed["system"][0]["cache_control"] == {"type": "ephemeral"}, observed
    assert len(observed["messages"][-1]["content"][0]["content"]) == 3, observed
    _stored(caller, f"msg_{marker}")


@pytest.mark.parametrize("shape", ("valid", "unchanged", "malformed", "duplicate"))
def test_tool_argument_compatibility(caller: Caller, shape: str) -> None:
    marker: Final = _identity()
    original: Final = {
        "valid": json.dumps(_input()),
        "unchanged": '{ "path" : "C:\\\\tmp", "count": 3, "ok": false }',
        "malformed": '{"env":"' + TEXT + '",broken',
        "duplicate": '{"env":"' + TEXT + '","env":"safe"}',
    }[shape]
    body: Final = _body("openai", marker)
    body["messages"][1]["tool_calls"][0]["function"]["arguments"] = original
    response: Final = caller.rig.proxy.request("POST", "/v1/chat/completions", body, key=caller.key)
    assert response.status_code == 200 and "accepted" in response.text, response.text
    selected: Final = tuple(request for request in caller.rig.wire.drain() if marker.encode() in request.body)
    assert len(selected) == 1, selected
    observed: Final = json.loads(selected[0].body)["messages"][1]["tool_calls"][0]["function"]["arguments"]
    if shape == "unchanged":
        assert observed == original, observed
    else:
        assert SECRET not in observed and "REDACTED" in observed, observed
    if shape == "valid":
        parsed: Final = json.loads(observed)
        assert parsed["path"] == _input()["path"] and parsed["count"] == 3, parsed
        assert parsed["enabled"] is False and parsed["optional"] is None, parsed
    _stored(caller, response.json()["id"], masked=shape != "unchanged")


@pytest.mark.parametrize(
    ("original", "masked"),
    (
        (
            '{ "env" : "' + TEXT + '", "amount": 0.123456789012345678901, "price": 1.50, "limit": 1e5 }',
            '{ "env" : "AWS_ACCESS_KEY_ID=[REDACTED]", "amount": 0.123456789012345678901, "price": 1.50, "limit": 1e5 }',
        ),
        ('{"password": "Zx9qL2mN8vB4\\"kR7Tq", "path": ".env"}', '{"password": "[REDACTED]\\"kR7Tq", "path": ".env"}'),
    ),
    ids=("number-literals", "keyword-match-inside-escape"),
)
def test_masked_tool_arguments_keep_their_other_text(caller: Caller, original: str, masked: str) -> None:
    marker: Final = _identity()
    body: Final = _body("openai", marker)
    body["messages"][1]["tool_calls"][0]["function"]["arguments"] = original
    response: Final = caller.rig.proxy.request("POST", "/v1/chat/completions", body, key=caller.key)
    assert response.status_code == 200 and "accepted" in response.text, response.text
    selected: Final = tuple(request for request in caller.rig.wire.drain() if marker.encode() in request.body)
    assert len(selected) == 1, selected
    observed: Final = json.loads(selected[0].body)["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert observed == masked, observed
    _stored(caller, response.json()["id"])


@pytest.mark.parametrize("path", ("/v1/responses", "/v1/completions"), ids=("responses", "plain-completion"))
def test_existing_text_shapes_still_redact(caller: Caller, path: str) -> None:
    marker: Final = _identity()
    body: Final = {
        "model": "gpt",
        "cache": {"no-cache": True},
        **({"input": f"{marker} {TEXT}"} if path.endswith("responses") else {"prompt": f"{marker} {TEXT}"}),
    }
    response: Final = caller.rig.proxy.request(
        "POST", path, body, key=caller.key, headers={"x-litellm-call-id": marker}
    )
    assert response.status_code == 200 and "accepted" in response.text, response.text
    _wire(caller, marker)
    request_id: Final = marker if path.endswith("responses") else response.json()["id"]
    _stored(caller, request_id)


_RESPONSES_ITEMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "instructions": (),
    "function-call": (
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "write_file",
            "arguments": json.dumps({"path": ".env", "content": TEXT}),
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "written"},
    ),
    "custom-tool-call": (
        {"type": "custom_tool_call", "call_id": "call_1", "name": "shell", "input": f"export {TEXT}"},
        {"type": "custom_tool_call_output", "call_id": "call_1", "output": "done"},
    ),
}


@pytest.mark.parametrize("shape", tuple(_RESPONSES_ITEMS))
def test_responses_tool_calls_and_instructions_are_redacted(caller: Caller, shape: str) -> None:
    marker: Final = _identity()
    body: Final = {
        "model": "gpt",
        "cache": {"no-cache": True},
        "input": [{"role": "user", "content": f"{marker} write the env file"}, *_RESPONSES_ITEMS[shape]],
        **({"instructions": TEXT} if shape == "instructions" else {}),
    }
    response: Final = caller.rig.proxy.request(
        "POST", "/v1/responses", body, key=caller.key, headers={"x-litellm-call-id": marker}
    )
    assert response.status_code == 200 and "accepted" in response.text, response.text
    sent: Final = _wire(caller, marker)
    if shape == "function-call":
        assert json.loads(sent["input"][1]["arguments"]) == {"path": ".env", "content": "AWS_ACCESS_KEY_ID=[REDACTED]"}
    if shape == "custom-tool-call":
        assert sent["input"][1]["input"] == "export AWS_ACCESS_KEY_ID=[REDACTED]"
    _stored(caller, marker)


def test_cache_hit_keeps_masked_guardrail_metadata(caller: Caller) -> None:
    marker: Final = _identity()
    body: Final = {**_body("openai", marker), "cache": {"ttl": 60}}
    first: Final = caller.rig.proxy.request("POST", "/v1/chat/completions", body, key=caller.key)
    assert first.status_code == 200, first.text
    _wire(caller, marker)
    _stored(caller, first.json()["id"])
    second: Final = caller.rig.proxy.request("POST", "/v1/chat/completions", body, key=caller.key)
    assert second.status_code == 200 and second.json()["id"] == first.json()["id"], second.text
    assert not tuple(request for request in caller.rig.wire.drain() if marker.encode() in request.body)
    _stored(caller, first.json()["id"], cache=True)


def test_key_permission_can_disable_default_guardrail(caller: Caller) -> None:
    marker: Final = _identity()
    with _key(caller.rig, permissions={"hide_secrets": False}) as key:
        response: Final = caller.rig.proxy.request("POST", "/v1/chat/completions", _body("openai", marker), key=key)
        assert response.status_code == 200, response.text
        selected: Final = tuple(request for request in caller.rig.wire.drain() if marker.encode() in request.body)
        assert len(selected) == 1 and SECRET.encode() in selected[0].body, selected
        _stored(caller, response.json()["id"], masked=False)


@pytest.mark.parametrize("provider", ("anthropic", "openai"))
def test_provider_error_occurs_after_redaction(caller: Caller, provider: str) -> None:
    marker: Final = _identity()
    body: Final = _body(provider, marker)
    body["messages"][0]["content"] += " provider-error"
    path: Final = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
    response: Final = caller.rig.proxy.request(
        "POST", path, body, key=caller.key, headers={"x-litellm-call-id": marker}
    )
    assert response.status_code == 401 and "scripted deployment refused" in response.text, response.text
    _wire(caller, marker)
    row: Final = _stored(caller, response.headers["x-litellm-call-id"])
    assert row["status"] == "failure", row


@pytest.mark.parametrize("problem", ("unauthenticated", "messages-int", "messages-list", "system-int"))
def test_invalid_requests_preserve_redaction_and_proxy_health(caller: Caller, problem: str) -> None:
    marker: Final = _identity()
    body: Final = _body("anthropic", marker)
    if problem == "messages-int":
        body["messages"] = 17
    elif problem == "messages-list":
        body["messages"] = [17]
    elif problem == "system-int":
        body["system"] = 17
    body["metadata"] = {"user_id": marker}
    response: Final = caller.rig.proxy.request(
        "POST",
        "/v1/messages",
        body,
        key="sk-deliberately-unknown" if problem == "unauthenticated" else caller.key,
    )
    assert response.status_code >= 400 and "error" in response.text.lower(), response.text
    selected: Final = tuple(request for request in caller.rig.wire.drain() if marker.encode() in request.body)
    if problem == "unauthenticated":
        assert not selected, selected
    else:
        assert all(SECRET.encode() not in request.body for request in selected), selected
    health: Final = caller.rig.proxy.request("GET", "/health/liveliness")
    assert health.status_code == 200, health.text


def test_surviving_worker_redacts_mixed_burst_and_replacement_recovers(isolated_caller: Caller) -> None:
    caller: Final = isolated_caller
    workers: Final = caller.rig.owned.worker_pids()
    assert len(workers) == 2, workers
    markers: Final = tuple(_identity() for _ in range(20))

    def send(item: tuple[int, str]) -> tuple[str, str]:
        index, marker = item
        provider: Final = "anthropic" if index % 2 else "openai"
        path: Final = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
        proxy: Final = caller.rig.proxy
        # a new connection per call: a pooled keep-alive socket to the killed worker can outlive the kill
        with httpx.Client(base_url=proxy.client.base_url, timeout=case_timeout_seconds(), trust_env=False) as client:
            response: Final = Gateway(client, proxy.key, proxy.upstream_url).request(
                "POST", path, _body(provider, marker, stream=index % 3 == 0), key=caller.key
            )
        assert response.status_code == 200 and "accepted" in response.text, response.text
        return marker, f"{'msg' if provider == 'anthropic' else 'chatcmpl'}_{marker}"

    with caller.rig.owned.replacing_worker(min(workers)) as kill:
        kill()
        with ThreadPoolExecutor(max_workers=20) as pool:
            results: Final = tuple(pool.map(send, enumerate(markers)))
    captured: Final = caller.rig.wire.drain()
    for marker, response_id in results:
        selected: Final = tuple(request for request in captured if marker.encode() in request.body)
        assert len(selected) == 1 and SECRET.encode() not in selected[0].body, selected
        _stored(caller, response_id)
    assert len(caller.rig.owned.worker_pids()) == 2 and min(workers) not in caller.rig.owned.worker_pids()
    recovery: Final = _identity()
    response: Final = caller.rig.proxy.request(
        "POST", "/v1/chat/completions", _body("openai", recovery), key=caller.key
    )
    assert response.status_code == 200, response.text
    _wire(caller, recovery)
    _stored(caller, response.json()["id"])
