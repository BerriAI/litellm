import json
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from pydantic import BaseModel, JsonValue, TypeAdapter
from redis import Redis

from tests.integration._support.client import Gateway, delete_key_if_present, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_AUDIENCE: Final = "litellm-integration"
_CLIENT_CLAIM: Final = "client_id"
_KEY_ID: Final = "integration-signing-key"
_UPSTREAM_KEY: Final = "synthetic-openai-key"
_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
_CHAT_BODY: Final = TypeAdapter(dict[str, JsonValue])
_EXPECTED_MODEL_DISCOVERY: Final = ("GET", "/v1/models")
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


class _KeyResponse(BaseModel):
    key: str


class _DeleteResponse(BaseModel):
    deleted_keys: tuple[str, ...]


class _JWTMappingResponse(BaseModel):
    id: str
    jwt_issuer: str | None
    jwt_claim_name: str
    jwt_claim_value: str
    is_active: bool


class _JWTMappingRow(BaseModel):
    id: str
    token: str
    jwt_issuer: str
    is_active: bool


class _SpendRow(BaseModel):
    api_key: str


class _CountRow(BaseModel):
    count: int


class _MappingIdRow(BaseModel):
    id: str


class _ModelEntry(BaseModel):
    id: str


class _ModelList(BaseModel):
    data: tuple[_ModelEntry, ...]


class _ProxyError(BaseModel):
    message: str
    type: str
    param: str
    code: str


class _ErrorResponse(BaseModel):
    error: _ProxyError


def _proxy_config(
    directory: Path,
    model: str,
    upstream_url: str,
    issuers: tuple[dict[str, JsonValue], ...] = (),
) -> Path:
    config: Final = directory / "jwt_key_mapping_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": (
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": f"openai/{model}",
                            "api_base": f"{upstream_url}/v1",
                            "api_key": _UPSTREAM_KEY,
                        },
                    },
                ),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "enable_jwt_auth": True,
                    "litellm_jwtauth": {
                        "user_id_jwt_field": "sub",
                        "virtual_key_claim_field": _CLIENT_CLAIM,
                        "unregistered_jwt_client_behavior": "reject",
                        **({"issuers": issuers} if issuers else {}),
                    },
                },
                "litellm_settings": {
                    "enable_redis_auth_cache": True,
                    "cache": True,
                    "cache_params": {
                        "type": "redis",
                        "host": "os.environ/REDIS_HOST",
                        "port": "os.environ/REDIS_PORT",
                    },
                },
            }
        )
    )
    return config


def _issuer_config(issuer: str, jwks_url: str) -> dict[str, JsonValue]:
    return {
        "issuer": issuer,
        "jwks_url": jwks_url,
        "audience": _AUDIENCE,
        "virtual_key_claim_field": _CLIENT_CLAIM,
        "user_id_jwt_field": "sub",
    }


def _jwks_body(private_key: rsa.RSAPrivateKey) -> bytes:
    public_jwk: Final = _CHAT_BODY.validate_json(RSAAlgorithm.to_jwk(private_key.public_key()))
    return json.dumps({"keys": [{**public_jwk, "kid": _KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()


def _jwks_respond(private_key: rsa.RSAPrivateKey) -> Callable[[Request], Reply]:
    body: Final = _jwks_body(private_key)

    def respond(request: Request) -> Reply:
        assert request.method == "GET"
        assert request.target == "/jwks"
        return Reply(body=body)

    return respond


def _signed_jwt(private_key: rsa.RSAPrivateKey, client_id: str, issuer: str | None = None) -> str:
    now: Final = int(time.time())
    claims: Final[dict[str, JsonValue]] = {
        "sub": "integration-user-" + uuid.uuid4().hex,
        "client_id": client_id,
        "aud": _AUDIENCE,
        "iat": now,
        "exp": now + 300,
        **({"iss": issuer} if issuer is not None else {}),
    }
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": _KEY_ID})


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
                "model": "integration-jwt-key-mapping",
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


def _chat_respond(backend: str, prompts: tuple[str, ...]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target == "/v1/models", request
            assert request.headers["authorization"] == f"Bearer {_UPSTREAM_KEY}"
            assert request.body == b""
            return Reply(
                body=json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": backend,
                                "object": "model",
                                "created": 0,
                                "owned_by": "integration",
                            }
                        ],
                    }
                ).encode()
            )
        assert request.method == "POST", request
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_UPSTREAM_KEY}"
        body: Final = _CHAT_BODY.validate_json(request.body)
        assert body in tuple(_chat_body(backend, prompt) for prompt in prompts), body
        return _wire_reply("chatcmpl-" + uuid.uuid4().hex)

    return respond


def _chat(gateway: Gateway, model: str, token: str, prompt: str) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", _chat_body(model, prompt), key=token)


def _assert_upstream_requests(upstream: Wire, expected: tuple[tuple[str, str], ...]) -> None:
    observed: Final = tuple(sorted((request.method, request.target) for request in upstream.drain()))
    assert observed == tuple(sorted(expected)), observed


def _assert_chat(gateway: Gateway, model: str, token: str, prompt: str, key_hash: str) -> _ChatResponse:
    response: Final = _chat(gateway, model, token, prompt)
    assert response.status_code == 200, response.text
    payload: Final = _ChatResponse.model_validate_json(response.content)
    assert payload.choices[0].message.content == "wire response", response.text
    assert payload.usage == _Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40), response.text
    spend: Final = TypeAdapter(tuple[_SpendRow, ...]).validate_python(
        eventually(
            lambda: read_rows(
                'SELECT api_key FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (payload.id,),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
    )
    assert spend == (_SpendRow(api_key=key_hash),)
    return payload


def _model_is_available(gateway: Gateway, model: str) -> bool:
    response: Final = gateway.request("GET", "/v1/models")
    if response.status_code != 200:
        return False
    payload: Final = _ModelList.model_validate_json(response.content)
    return any(entry.id == model for entry in payload.data)


def _create_key(gateway: Gateway, model: str) -> str:
    response: Final = gateway.request("POST", "/key/generate", {"models": [model]})
    assert response.status_code == 200, response.text
    return _KeyResponse.model_validate_json(response.content).key


def _create_mapping(
    gateway: Gateway,
    key: str,
    client_id: str,
    issuer: str | None = None,
    spelling: Literal["key", "token"] = "key",
) -> _JWTMappingResponse:
    body: Final = {
        "jwt_claim_name": _CLIENT_CLAIM,
        "jwt_claim_value": client_id,
        spelling: key if spelling == "key" else sha256(key.encode()).hexdigest(),
        **({"jwt_issuer": issuer} if issuer is not None else {}),
    }
    response: Final = gateway.request("POST", "/jwt/key/mapping/new", body)
    assert response.status_code == 200, response.text
    return _JWTMappingResponse.model_validate_json(response.content)


def _delete_mapping_if_present(gateway: Gateway, identity: str) -> None:
    rows: Final = TypeAdapter(tuple[_MappingIdRow, ...]).validate_python(
        read_rows('SELECT id FROM "LiteLLM_JWTKeyMapping" WHERE id=%s', (identity,))
    )
    if not rows:
        return
    response: Final = gateway.request("POST", "/jwt/key/mapping/delete", {"id": identity})
    assert response.status_code == 200, response.text


def _assert_unmapped(response: httpx.Response, client_id: str) -> None:
    assert response.status_code == 403, response.text
    error: Final = _ErrorResponse.model_validate_json(response.content)
    expected: Final = f"JWT Key Mapping: No registered mapping for {_CLIENT_CLAIM}='{client_id}'. Access denied."
    assert error.error == _ProxyError(message=expected, type="auth_error", param="None", code="403"), response.text


def _mapping_info(gateway: Gateway, identity: str) -> _JWTMappingResponse:
    response: Final = gateway.request("GET", "/jwt/key/mapping/info", params={"id": identity})
    assert response.status_code == 200, response.text
    return _JWTMappingResponse.model_validate_json(response.content)


def _mapping_row(gateway: Gateway, identity: str) -> _JWTMappingRow:
    rows: Final = TypeAdapter(tuple[_JWTMappingRow, ...]).validate_python(
        read_rows(
            'SELECT id, token, jwt_issuer, is_active FROM "LiteLLM_JWTKeyMapping" WHERE id=%s',
            (identity,),
        )
    )
    assert len(rows) == 1
    return rows[0]


def _spend_count(gateway: Gateway, key_hash: str) -> int:
    rows: Final = TypeAdapter(tuple[_CountRow, ...]).validate_python(
        read_rows('SELECT count(*) AS count FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (key_hash,))
    )
    assert len(rows) == 1
    return rows[0].count


@contextmanager
def _two_workers(
    gateway: Gateway, tmp_path: Path, config: Path, overrides: Mapping[str, str]
) -> Iterator[tuple[Gateway, Gateway]]:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        baseline: Final = cache.pubsub_numsub(_CHANNEL)[0][1]
        with ExitStack() as stack:
            first: Final = stack.enter_context(
                owned_proxy(gateway, tmp_path / "worker-one", dict(overrides), config=config)
            )
            second: Final = stack.enter_context(
                owned_proxy(gateway, tmp_path / "worker-two", dict(overrides), config=config)
            )
            subscribers: Final = eventually(
                lambda: cache.pubsub_numsub(_CHANNEL)[0][1],
                lambda count: count >= baseline + 2,
                seconds=20,
            )
            assert subscribers >= baseline + 2
            yield first, second


def test_updating_a_warmed_mapping_deactivates_and_repoints_it_on_both_workers(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    model: Final = "jwt-key-mapping-" + uuid.uuid4().hex
    backend: Final = model
    client_id: Final = "integration-client-" + uuid.uuid4().hex
    prompts: Final = (
        "mapping warm primary " + uuid.uuid4().hex,
        "mapping warm peer " + uuid.uuid4().hex,
        "mapping inactive primary " + uuid.uuid4().hex,
        "mapping inactive peer " + uuid.uuid4().hex,
        "mapping repointed primary " + uuid.uuid4().hex,
        "mapping repointed peer " + uuid.uuid4().hex,
        "mapping restored primary " + uuid.uuid4().hex,
        "mapping restored peer " + uuid.uuid4().hex,
    )
    token: Final = _signed_jwt(private_key, client_id)

    with wire_server(_chat_respond(backend, prompts)) as upstream, wire_server(_jwks_respond(private_key)) as jwks:
        config: Final = _proxy_config(tmp_path, model, upstream.url)
        overrides: Final = {"JWT_PUBLIC_KEY_URL": f"{jwks.url}/jwks", "JWT_AUDIENCE": _AUDIENCE}
        with _two_workers(gateway, tmp_path, config, overrides) as (first, second):
            eventually(lambda: _model_is_available(first, model), bool, seconds=30)
            eventually(lambda: _model_is_available(second, model), bool, seconds=30)
            with first.scenario() as scenario:
                old_key: Final = _create_key(first, model)
                new_key: Final = _create_key(first, model)
                scenario.cleanups.callback(delete_key_if_present, first, old_key)
                scenario.cleanups.callback(delete_key_if_present, first, new_key)
                old_hash: Final = sha256(old_key.encode()).hexdigest()
                new_hash: Final = sha256(new_key.encode()).hexdigest()
                mapping: Final = _create_mapping(first, old_key, client_id)
                scenario.cleanups.callback(_delete_mapping_if_present, first, mapping.id)

                first_response: Final = _assert_chat(first, model, token, prompts[0], old_hash)
                second_response: Final = _assert_chat(second, model, token, prompts[1], old_hash)
                assert first_response.id != second_response.id
                _assert_upstream_requests(upstream, (_EXPECTED_MODEL_DISCOVERY,) * 2 + (_EXPECTED_CHAT,) * 2)

                deactivated_response: Final = first.request(
                    "POST", "/jwt/key/mapping/update", {"id": mapping.id, "is_active": False}
                )
                assert deactivated_response.status_code == 200, deactivated_response.text
                deactivated: Final = _JWTMappingResponse.model_validate_json(deactivated_response.content)
                assert deactivated.is_active is False, deactivated_response.text
                assert _mapping_info(first, mapping.id).is_active is False
                inactive_row: Final = _mapping_row(first, mapping.id)
                assert inactive_row.token == old_hash and inactive_row.is_active is False
                old_spend_count: Final = _spend_count(first, old_hash)

                refused_first: Final = _chat(first, model, token, prompts[2])
                _assert_unmapped(refused_first, client_id)
                refused_second: Final = eventually(
                    lambda: _chat(second, model, token, prompts[3]),
                    lambda response: response.status_code == 403,
                    seconds=3,
                )
                _assert_unmapped(refused_second, client_id)
                _assert_upstream_requests(upstream, ())
                assert _spend_count(first, old_hash) == old_spend_count

                reactivated_response: Final = first.request(
                    "POST",
                    "/jwt/key/mapping/update",
                    {"id": mapping.id, "is_active": True, "token": new_hash},
                )
                assert reactivated_response.status_code == 200, reactivated_response.text
                reactivated: Final = _JWTMappingResponse.model_validate_json(reactivated_response.content)
                assert reactivated.is_active is True, reactivated_response.text
                assert _mapping_info(first, mapping.id).is_active is True
                active_row: Final = _mapping_row(first, mapping.id)
                assert active_row.token == new_hash and active_row.is_active is True
                _assert_chat(first, model, token, prompts[4], new_hash)
                _assert_chat(second, model, token, prompts[5], new_hash)
                _assert_upstream_requests(upstream, (_EXPECTED_CHAT,) * 2)

                restored_response: Final = first.request(
                    "POST",
                    "/jwt/key/mapping/update",
                    {"id": mapping.id, "key": old_key},
                )
                assert restored_response.status_code == 200, restored_response.text
                restored: Final = _JWTMappingResponse.model_validate_json(restored_response.content)
                assert restored.is_active is True, restored_response.text
                assert _mapping_info(first, mapping.id).is_active is True
                restored_row: Final = _mapping_row(first, mapping.id)
                assert restored_row.token == old_hash and restored_row.is_active is True
                _assert_chat(first, model, token, prompts[6], old_hash)
                _assert_chat(second, model, token, prompts[7], old_hash)
                _assert_upstream_requests(upstream, (_EXPECTED_CHAT,) * 2)


@pytest.mark.parametrize("spelling", ("body", "path"))
def test_regenerating_a_jwt_mapped_key_moves_jwt_callers_to_the_new_hash(
    gateway: Gateway, tmp_path: Path, spelling: str
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    model: Final = "jwt-key-mapping-" + uuid.uuid4().hex
    backend: Final = model
    client_id: Final = "integration-client-" + uuid.uuid4().hex
    prompts: Final = (
        "regenerate warm primary " + uuid.uuid4().hex,
        "regenerate warm peer " + uuid.uuid4().hex,
        "regenerated primary " + uuid.uuid4().hex,
        "regenerated peer " + uuid.uuid4().hex,
    )
    token: Final = _signed_jwt(private_key, client_id)

    with wire_server(_chat_respond(backend, prompts)) as upstream, wire_server(_jwks_respond(private_key)) as jwks:
        config: Final = _proxy_config(tmp_path, model, upstream.url)
        overrides: Final = {"JWT_PUBLIC_KEY_URL": f"{jwks.url}/jwks", "JWT_AUDIENCE": _AUDIENCE}
        with _two_workers(gateway, tmp_path, config, overrides) as (first, second):
            eventually(lambda: _model_is_available(first, model), bool, seconds=30)
            eventually(lambda: _model_is_available(second, model), bool, seconds=30)
            with first.scenario() as scenario:
                old_key: Final = _create_key(first, model)
                scenario.cleanups.callback(delete_key_if_present, first, old_key)
                old_hash: Final = sha256(old_key.encode()).hexdigest()
                mapping: Final = _create_mapping(first, old_key, client_id)
                scenario.cleanups.callback(_delete_mapping_if_present, first, mapping.id)
                _assert_chat(first, model, token, prompts[0], old_hash)
                _assert_chat(second, model, token, prompts[1], old_hash)
                _assert_upstream_requests(upstream, (_EXPECTED_MODEL_DISCOVERY,) * 2 + (_EXPECTED_CHAT,) * 2)

                regenerated_response: Final = (
                    first.request("POST", "/key/regenerate", {"key": old_key})
                    if spelling == "body"
                    else first.request("POST", f"/key/{old_hash}/regenerate", {})
                )
                assert regenerated_response.status_code == 200, regenerated_response.text
                new_key: Final = _KeyResponse.model_validate_json(regenerated_response.content).key
                scenario.cleanups.callback(delete_key_if_present, first, new_key)
                new_hash: Final = sha256(new_key.encode()).hexdigest()
                mapped_row: Final = _mapping_row(first, mapping.id)
                assert mapped_row.token == new_hash and mapped_row.is_active is True
                _assert_chat(first, model, token, prompts[2], new_hash)
                _assert_chat(second, model, token, prompts[3], new_hash)
                _assert_upstream_requests(upstream, (_EXPECTED_CHAT,) * 2)


def test_deleting_a_jwt_mapped_key_stops_its_mapping_on_both_workers(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    model: Final = "jwt-key-mapping-" + uuid.uuid4().hex
    backend: Final = model
    client_id: Final = "integration-client-" + uuid.uuid4().hex
    prompts: Final = (
        "delete mapped warm primary " + uuid.uuid4().hex,
        "delete mapped warm peer " + uuid.uuid4().hex,
        "delete mapped primary " + uuid.uuid4().hex,
        "delete mapped peer " + uuid.uuid4().hex,
    )
    token: Final = _signed_jwt(private_key, client_id)

    with wire_server(_chat_respond(backend, prompts)) as upstream, wire_server(_jwks_respond(private_key)) as jwks:
        config: Final = _proxy_config(tmp_path, model, upstream.url)
        overrides: Final = {"JWT_PUBLIC_KEY_URL": f"{jwks.url}/jwks", "JWT_AUDIENCE": _AUDIENCE}
        with _two_workers(gateway, tmp_path, config, overrides) as (first, second):
            eventually(lambda: _model_is_available(first, model), bool, seconds=30)
            eventually(lambda: _model_is_available(second, model), bool, seconds=30)
            with first.scenario() as scenario:
                key: Final = _create_key(first, model)
                scenario.cleanups.callback(delete_key_if_present, first, key)
                key_hash: Final = sha256(key.encode()).hexdigest()
                mapping: Final = _create_mapping(first, key, client_id)
                scenario.cleanups.callback(_delete_mapping_if_present, first, mapping.id)
                warm_first: Final = _assert_chat(first, model, token, prompts[0], key_hash)
                warm_second: Final = _assert_chat(second, model, token, prompts[1], key_hash)
                assert warm_first.id != warm_second.id
                _assert_upstream_requests(upstream, (_EXPECTED_MODEL_DISCOVERY,) * 2 + (_EXPECTED_CHAT,) * 2)
                warm_spend_count: Final = _spend_count(first, key_hash)
                assert warm_spend_count == 2

                deletion: Final = first.request("POST", "/key/delete", {"keys": [key]})
                assert deletion.status_code == 200, deletion.text
                assert _DeleteResponse.model_validate_json(deletion.content).deleted_keys == (key,), deletion.text
                assert read_rows('SELECT id FROM "LiteLLM_JWTKeyMapping" WHERE id=%s', (mapping.id,)) == []
                first_refusal: Final = _chat(first, model, token, prompts[2])
                _assert_unmapped(first_refusal, client_id)
                second_refusal: Final = eventually(
                    lambda: _chat(second, model, token, prompts[3]),
                    lambda response: response.status_code == 403,
                    seconds=3,
                )
                _assert_unmapped(second_refusal, client_id)
                _assert_upstream_requests(upstream, ())
                assert _spend_count(first, key_hash) == warm_spend_count


def test_issuer_scoped_mappings_with_the_same_claim_resolve_per_issuer(gateway: Gateway, tmp_path: Path) -> None:
    private_key_a: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_b: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    model: Final = "jwt-key-mapping-" + uuid.uuid4().hex
    backend: Final = model
    client_id: Final = "integration-client-" + uuid.uuid4().hex
    issuer_a: Final = "https://idp-a.integration.example"
    issuer_b: Final = "https://idp-b.integration.example"
    prompts: Final = tuple(f"issuer mapping {index} {uuid.uuid4().hex}" for index in range(4))

    with (
        wire_server(_chat_respond(backend, prompts)) as upstream,
        wire_server(_jwks_respond(private_key_a)) as jwks_a,
        wire_server(_jwks_respond(private_key_b)) as jwks_b,
    ):
        issuer_configurations: Final = (
            _issuer_config(issuer_a, f"{jwks_a.url}/jwks"),
            _issuer_config(issuer_b, f"{jwks_b.url}/jwks"),
        )
        config: Final = _proxy_config(tmp_path, model, upstream.url, issuer_configurations)
        with owned_proxy(gateway, tmp_path / "issuer-worker", {}, config=config) as candidate:
            eventually(lambda: _model_is_available(candidate, model), bool, seconds=30)
            with candidate.scenario() as scenario:
                key_a: Final = _create_key(candidate, model)
                key_b: Final = _create_key(candidate, model)
                scenario.cleanups.callback(delete_key_if_present, candidate, key_a)
                scenario.cleanups.callback(delete_key_if_present, candidate, key_b)
                hash_a: Final = sha256(key_a.encode()).hexdigest()
                hash_b: Final = sha256(key_b.encode()).hexdigest()
                mapping_a: Final = _create_mapping(candidate, key_a, client_id, issuer_a)
                mapping_b: Final = _create_mapping(candidate, key_b, client_id, issuer_b, spelling="token")
                scenario.cleanups.callback(_delete_mapping_if_present, candidate, mapping_a.id)
                scenario.cleanups.callback(_delete_mapping_if_present, candidate, mapping_b.id)

                _assert_chat(candidate, model, _signed_jwt(private_key_a, client_id, issuer_a), prompts[0], hash_a)
                _assert_chat(candidate, model, _signed_jwt(private_key_b, client_id, issuer_b), prompts[1], hash_b)
                _assert_chat(candidate, model, _signed_jwt(private_key_a, client_id, issuer_a), prompts[2], hash_a)
                _assert_chat(candidate, model, _signed_jwt(private_key_b, client_id, issuer_b), prompts[3], hash_b)
                rows: Final = TypeAdapter(tuple[_JWTMappingRow, ...]).validate_python(
                    read_rows(
                        'SELECT id, token, jwt_issuer, is_active FROM "LiteLLM_JWTKeyMapping" '
                        "WHERE jwt_claim_name=%s AND jwt_claim_value=%s ORDER BY jwt_issuer",
                        (_CLIENT_CLAIM, client_id),
                    )
                )
                assert rows == (
                    _JWTMappingRow(id=mapping_a.id, token=hash_a, jwt_issuer=issuer_a, is_active=True),
                    _JWTMappingRow(id=mapping_b.id, token=hash_b, jwt_issuer=issuer_b, is_active=True),
                )
        _assert_upstream_requests(upstream, (_EXPECTED_MODEL_DISCOVERY,) + (_EXPECTED_CHAT,) * 4)
        requests_a: Final = jwks_a.drain()
        requests_b: Final = jwks_b.drain()
        assert requests_a and all((request.method, request.target) == ("GET", "/jwks") for request in requests_a)
        assert requests_b and all((request.method, request.target) == ("GET", "/jwks") for request in requests_b)
