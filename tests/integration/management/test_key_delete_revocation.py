import json
import uuid
from hashlib import sha256
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, delete_key_if_present, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, wire_server

_CHAT_BODY: Final = TypeAdapter(dict[str, JsonValue])
_EXPECTED_CHAT: Final = ("POST", "/v1/chat/completions")


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _AssistantMessage(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _AssistantMessage


class _ChatResponse(BaseModel):
    id: str
    choices: tuple[_Choice, ...]
    usage: _Usage


class _ModelEntry(BaseModel):
    id: str


class _ModelList(BaseModel):
    data: tuple[_ModelEntry, ...]


class _GenerateKeyResponse(BaseModel):
    key: str


class _DeleteResponse(BaseModel):
    deleted_keys: tuple[str, ...]


class _KeyInfo(BaseModel):
    status: str


class _KeyInfoResponse(BaseModel):
    info: _KeyInfo


class _Error(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _Error


class _DeletedKeyRow(BaseModel):
    token: str


def _chat_body(model: str, prompt: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }


def _wire_reply(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "integration-keys-auth-delete",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "wire response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
            }
        ).encode()
    )


def _chat(gateway: Gateway, model: str, key: str, prompt: str) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", _chat_body(model, prompt), key=key)


def _assert_success(response: httpx.Response) -> _ChatResponse:
    assert response.status_code == 200, response.text
    payload: Final = _ChatResponse.model_validate_json(response.content)
    assert payload.choices[0].message.content == "wire response", response.text
    assert payload.usage == _Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40), response.text
    return payload


def _model_is_available(gateway: Gateway, model: str) -> bool:
    response: Final = gateway.request("GET", "/v1/models")
    if response.status_code != 200:
        return False
    models: Final = _ModelList.model_validate_json(response.content)
    return model in tuple(entry.id for entry in models.data)


@pytest.mark.parametrize("spelling", ("keys", "hashed_keys", "key_aliases"))
def test_deleting_a_key_revokes_it(gateway: Gateway, spelling: str) -> None:
    backend: Final = "integration-keys-auth-delete"
    prompts: Final = (
        "delete warm " + uuid.uuid4().hex,
        "delete reject " + uuid.uuid4().hex,
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = _CHAT_BODY.validate_json(request.body)
        assert body in tuple(_chat_body(backend, prompt) for prompt in prompts), body
        return _wire_reply("chatcmpl-" + uuid.uuid4().hex)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{backend}", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        eventually(lambda: _model_is_available(gateway, model), bool, seconds=30)
        alias: Final = "integration-delete-" + uuid.uuid4().hex
        created_response: Final = gateway.request("POST", "/key/generate", {"key_alias": alias, "models": [model]})
        assert created_response.status_code == 200, created_response.text
        key: Final = _GenerateKeyResponse.model_validate_json(created_response.content).key
        scenario.cleanups.callback(delete_key_if_present, gateway, key)
        key_hash: Final = sha256(key.encode()).hexdigest()
        _assert_success(_chat(gateway, model, key, prompts[0]))
        assert tuple((request.method, request.target) for request in wire.drain()) == (_EXPECTED_CHAT,)

        identifier: Final = {"keys": key, "hashed_keys": key_hash, "key_aliases": alias}[spelling]
        delete_body: Final = {"key_aliases" if spelling == "key_aliases" else "keys": [identifier]}
        deleted: Final = gateway.request("POST", "/key/delete", delete_body)
        assert deleted.status_code == 200, deleted.text
        assert _DeleteResponse.model_validate_json(deleted.content).deleted_keys == (identifier,), deleted.text
        primary_refusal: Final = _chat(gateway, model, key, prompts[1])
        assert primary_refusal.status_code == 401, primary_refusal.text
        assert _ErrorResponse.model_validate_json(primary_refusal.content).error.type == "token_not_found_in_db", (
            primary_refusal.text
        )
        assert wire.drain() == (), "deleted-key requests reached the upstream"
        info: Final = gateway.request("GET", "/key/info", params={"key": key_hash})
        assert info.status_code == 200, info.text
        assert _KeyInfoResponse.model_validate_json(info.content).info.status == "deleted", info.text
        live_rows: Final = read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (key_hash,))
        deleted_rows: Final = TypeAdapter(tuple[_DeletedKeyRow, ...]).validate_python(
            read_rows('SELECT token FROM "LiteLLM_DeletedVerificationToken" WHERE token=%s', (key_hash,))
        )
        assert live_rows == []
        assert deleted_rows == (_DeletedKeyRow(token=key_hash),)


def test_deleting_a_key_revokes_it_on_a_warmed_peer(gateway: Gateway, peer: Gateway) -> None:
    pytest.skip(
        "BUG: POST /key/delete does not broadcast the key eviction, so a warmed peer keeps answering 200 for about "
        "60 seconds until its in-memory cache TTL expires"
    )
    backend: Final = "integration-keys-auth-delete"
    prompts: Final = (
        "delete warm primary " + uuid.uuid4().hex,
        "delete warm peer " + uuid.uuid4().hex,
        "delete reject peer " + uuid.uuid4().hex,
        "delete reject peer fresh " + uuid.uuid4().hex,
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = _CHAT_BODY.validate_json(request.body)
        assert body in tuple(_chat_body(backend, prompt) for prompt in prompts), body
        return _wire_reply("chatcmpl-" + uuid.uuid4().hex)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{backend}", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        eventually(lambda: _model_is_available(peer, model), bool, seconds=30)
        alias: Final = "integration-delete-" + uuid.uuid4().hex
        created_response: Final = gateway.request("POST", "/key/generate", {"key_alias": alias, "models": [model]})
        assert created_response.status_code == 200, created_response.text
        key: Final = _GenerateKeyResponse.model_validate_json(created_response.content).key
        scenario.cleanups.callback(delete_key_if_present, gateway, key)
        warm_responses: Final = tuple(
            _assert_success(_chat(worker, model, key, prompt))
            for worker, prompt in zip((gateway, peer), prompts[:2], strict=True)
        )
        assert len(warm_responses) == 2
        assert tuple((request.method, request.target) for request in wire.drain()) == (_EXPECTED_CHAT,) * 2
        deleted: Final = gateway.request("POST", "/key/delete", {"keys": [key]})
        assert deleted.status_code == 200, deleted.text
        assert _DeleteResponse.model_validate_json(deleted.content).deleted_keys == (key,), deleted.text
        peer_refusal: Final = eventually(
            lambda: _chat(peer, model, key, prompts[2]),
            lambda response: response.status_code == 401,
            seconds=3,
        )
        assert _ErrorResponse.model_validate_json(peer_refusal.content).error.type == "token_not_found_in_db", (
            peer_refusal.text
        )
        wire.drain()
        fresh_peer_refusal: Final = _chat(peer, model, key, prompts[3])
        assert fresh_peer_refusal.status_code == 401, fresh_peer_refusal.text
        assert _ErrorResponse.model_validate_json(fresh_peer_refusal.content).error.type == "token_not_found_in_db", (
            fresh_peer_refusal.text
        )
        fresh_gateway_refusal: Final = _chat(gateway, model, key, prompts[3])
        assert fresh_gateway_refusal.status_code == 401, fresh_gateway_refusal.text
        assert _ErrorResponse.model_validate_json(fresh_gateway_refusal.content).error.type == (
            "token_not_found_in_db"
        ), fresh_gateway_refusal.text
        assert wire.drain() == (), "deleted-key requests reached the upstream"
