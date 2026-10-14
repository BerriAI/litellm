from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from integration.authorization._guardrail_opt_out import upstream_observations
from openai.types.chat import ChatCompletion
from pydantic import JsonValue, TypeAdapter
from redis import Redis

CONFIG_STORE_ID: Final = "vs_integration_config_store"
PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
REMOVE_OPENAI_API_BASE: Final = ("OPENAI_API_BASE",)
JWT_KEY_ID: Final = "integration-vector-store-jwt-key"
AUTH_CACHE_INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
JsonObject: TypeAlias = dict[str, JsonValue]
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _json_array(*values: JsonValue) -> JsonValue:
    return [*values]  # mutable-ok: request payloads and YAML sequences require list values


def _permission_for_stores(*store_ids: str) -> JsonObject:
    permission: Final[JsonObject] = {"vector_stores": _json_array(*store_ids)}
    return permission


def _key_for_scope(scenario: Scenario, model: str, scope: Literal["key", "team"], store_id: str) -> str:
    if scope == "key":
        return scenario.key(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    team: Final = scenario.team(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    return scenario.key(team_id=team, models=_json_array(model))


def _rag_query_body(model: str, marker: str, store_id: str, *, stream: bool | None = None) -> JsonObject:
    body: Final[JsonObject] = {
        "model": model,
        "messages": _json_array({"role": "user", "content": marker}),
        "retrieval_config": {"vector_store_id": store_id, "custom_llm_provider": "openai", "top_k": 1},
    }
    if stream is not None:
        body["stream"] = stream
    return body


def _rag_query(
    gateway: Gateway,
    model: str,
    marker: str,
    key: str,
    *,
    store_id: str = CONFIG_STORE_ID,
    path: str = "/v1/rag/query",
    stream: bool | None = None,
) -> httpx.Response:
    return gateway.request("POST", path, _rag_query_body(model, marker, store_id, stream=stream), key=key)


def _searches_for_marker(
    gateway: Gateway, marker: str, store_id: str = CONFIG_STORE_ID
) -> tuple[Mapping[str, JsonValue], ...]:
    search_path: Final = f"/vector_stores/{store_id}/search"
    return tuple(
        observation
        for observation in upstream_observations(gateway)
        if observation["path"] == search_path and marker in str(observation["body"])
    )


def _no_registry_config(directory: Path) -> Path:
    config: Final = object_value(yaml.safe_load(PROXY_CONFIG.read_text()))
    config_without_registry: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {name: value for name, value in config.items() if name != "vector_store_registry"}
    )
    yaml_config: Final[JsonObject] = {**config_without_registry, "model_list": _json_array()}
    path: Final = directory / "proxy_no_vector_store_registry.yaml"
    path.write_text(yaml.safe_dump(yaml_config))
    return path


def _openai_environment(gateway: Gateway) -> Mapping[str, str]:
    return MappingProxyType({"OPENAI_BASE_URL": gateway.upstream_url, "OPENAI_API_KEY": "synthetic-openai-key"})


@pytest.fixture(scope="module")
def no_registry_gateways(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Gateway]]:
    with gateway_from_environment() as upstream_gateway:
        directory: Final = tmp_path_factory.mktemp("rag_query_no_registry")
        config: Final = _no_registry_config(directory)
        with owned_proxy(
            upstream_gateway,
            directory,
            _openai_environment(upstream_gateway),
            config=config,
            remove_environment=REMOVE_OPENAI_API_BASE,
            workers=2,
        ) as no_registry_gateway:
            yield no_registry_gateway, upstream_gateway


class StrictGateway:
    def __init__(
        self,
        gateway: Gateway,
        upstream: Gateway,
        signing_key: rsa.RSAPrivateKey,
        config: Path,
        environment: Mapping[str, str],
    ) -> None:
        self.gateway: Final = gateway
        self.upstream: Final = upstream
        self._signing_key: Final = signing_key
        self.config: Final = config
        self.environment: Final = environment

    def jwt(self, subject: str, groups: tuple[str, ...] = ()) -> str:
        claims: Final[JsonObject] = {
            "sub": subject,
            "groups": _json_array(*groups),
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
        }
        return jwt.encode(claims, self._signing_key, algorithm="RS256", headers={"kid": JWT_KEY_ID})


def _strict_config(directory: Path, *, deny_by_default: bool = True) -> Path:
    config: Final = object_value(yaml.safe_load(PROXY_CONFIG.read_text()))
    general_settings: Final = object_value(config["general_settings"])
    strict: Final[JsonObject] = {
        **config,
        "general_settings": {
            **general_settings,
            "vector_store_deny_by_default": deny_by_default,
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "team_ids_jwt_field": "groups",
                "team_allowed_routes": _json_array("openai_routes", "info_routes", "/v1/rag/query"),
            },
        },
    }
    path: Final = directory / f"proxy_vector_store_deny_by_default_{deny_by_default}.yaml"
    path.write_text(yaml.safe_dump(strict))
    return path


@pytest.fixture(scope="module")
def strict_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[StrictGateway]:
    signing_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key())
    jwks_body: Final = json.dumps({"keys": [{**json.loads(public_jwk), "kid": JWT_KEY_ID}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=jwks_body)

    with gateway_from_environment() as upstream_gateway, wire_server(respond) as jwks:
        directory: Final = tmp_path_factory.mktemp("rag_query_deny_by_default")
        config: Final = _strict_config(directory)
        environment: Final = MappingProxyType({**_openai_environment(upstream_gateway), "JWT_PUBLIC_KEY_URL": jwks.url})
        with owned_proxy(
            upstream_gateway,
            directory,
            environment,
            config=config,
            remove_environment=REMOVE_OPENAI_API_BASE,
        ) as gateway:
            yield StrictGateway(gateway, upstream_gateway, signing_key, config, environment)


StrictCase: TypeAlias = Literal[
    "standalone_key_no_permission",
    "team_key_empty_key_grants",
    "team_key_empty_team_grants",
    "multi_store_one_ungranted",
    "jwt_user_without_grant",
]


def _strict_denied_request(
    strict: StrictGateway, scenario: Scenario, case: StrictCase, model: str, marker: str, store_id: str
) -> tuple[httpx.Response, str]:
    models: Final = _json_array(model)
    granted: Final = _permission_for_stores(store_id)
    empty: Final = _permission_for_stores()
    if case == "standalone_key_no_permission":
        standalone_key: Final = scenario.key(models=models)
        return strict.gateway.request(
            "POST", "/v1/rag/query", _rag_query_body(model, marker, store_id), key=standalone_key
        ), ("key_vector_store_access_denied")
    if case in ("team_key_empty_key_grants", "team_key_empty_team_grants"):
        key_grants_store: Final = case == "team_key_empty_team_grants"
        team: Final = scenario.team(models=models, object_permission=empty if key_grants_store else granted)
        team_key: Final = scenario.key(
            team_id=team, models=models, object_permission=granted if key_grants_store else empty
        )
        error_type: Final = "team_vector_store_access_denied" if key_grants_store else "key_vector_store_access_denied"
        return strict.gateway.request(
            "POST", "/v1/rag/query", _rag_query_body(model, marker, store_id), key=team_key
        ), (error_type)
    if case == "multi_store_one_ungranted":
        partial_key: Final = scenario.key(models=models, object_permission=_permission_for_stores(store_id))
        body: Final[JsonObject] = {
            **_rag_query_body(model, marker, store_id),
            "tools": _json_array({"type": "file_search", "vector_store_ids": _json_array(CONFIG_STORE_ID)}),
        }
        return strict.gateway.request(
            "POST", "/v1/chat/completions", body, key=partial_key
        ), "key_vector_store_access_denied"
    user: Final = scenario.user(user_role="internal_user")
    return (
        strict.gateway.request("POST", "/v1/rag/query", _rag_query_body(model, marker, store_id), key=strict.jwt(user)),
        "user_vector_store_access_denied",
    )


@pytest.mark.parametrize(
    "case",
    (
        "standalone_key_no_permission",
        "team_key_empty_key_grants",
        "team_key_empty_team_grants",
        "multi_store_one_ungranted",
        "jwt_user_without_grant",
    ),
)
def test_deny_by_default_rejects_ungranted_store_before_upstream_search(
    strict_gateway: StrictGateway, case: StrictCase
) -> None:
    with strict_gateway.gateway.scenario() as scenario:
        model: Final = scenario.model()
        store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
        marker: Final = f"lit6035 deny by default {case} {uuid.uuid4().hex}"

        response, error_type = _strict_denied_request(strict_gateway, scenario, case, model, marker, store_id)
        observations: Final = tuple(
            observation
            for observation in upstream_observations(strict_gateway.upstream)
            if marker in str(observation["body"])
        )

        assert response.status_code == 401, f"{response.text}; scripted_upstream_observations={observations!r}"
        assert response.json()["error"]["type"] == error_type, response.text
        assert observations == ()


GrantedCase: TypeAlias = Literal[
    "standalone_key_granted_registered_store",
    "team_key_both_grant_unregistered_store",
    "jwt_team_member_team_grant_only",
    "jwt_user_personal_grant",
    "master_key_without_grants",
]


def _strict_granted_request(
    strict: StrictGateway, scenario: Scenario, case: GrantedCase, model: str, marker: str, store_id: str
) -> httpx.Response:
    models: Final = _json_array(model)
    granted: Final = _permission_for_stores(store_id)
    body: Final = _rag_query_body(model, marker, store_id)
    if case in ("standalone_key_granted_registered_store", "team_key_both_grant_unregistered_store"):
        key_team: Final = scenario.team(models=models, object_permission=granted) if case.startswith("team") else None
        granted_key: Final = scenario.key(
            models=models, object_permission=granted, **({} if key_team is None else {"team_id": key_team})
        )
        return strict.gateway.request("POST", "/v1/rag/query", body, key=granted_key)
    if case == "jwt_team_member_team_grant_only":
        member_team: Final = scenario.team(models=models, object_permission=granted)
        member: Final = scenario.member(member_team)
        return strict.gateway.request("POST", "/v1/rag/query", body, key=strict.jwt(member, (member_team,)))
    if case == "jwt_user_personal_grant":
        user: Final = scenario.user(user_role="internal_user", object_permission=granted)
        return strict.gateway.request("POST", "/v1/rag/query", body, key=strict.jwt(user))
    return strict.gateway.request("POST", "/v1/rag/query", body)


@pytest.mark.parametrize(
    "case",
    (
        "standalone_key_granted_registered_store",
        "team_key_both_grant_unregistered_store",
        "jwt_team_member_team_grant_only",
        "jwt_user_personal_grant",
        "master_key_without_grants",
    ),
)
def test_deny_by_default_searches_explicitly_granted_store(strict_gateway: StrictGateway, case: GrantedCase) -> None:
    with strict_gateway.gateway.scenario() as scenario:
        model: Final = scenario.model()
        store_id: Final = (
            CONFIG_STORE_ID
            if case == "standalone_key_granted_registered_store"
            else f"vs_unregistered_{uuid.uuid4().hex}"
        )
        marker: Final = f"lit6035 granted {case} {uuid.uuid4().hex}"

        response: Final = _strict_granted_request(strict_gateway, scenario, case, model, marker, store_id)
        assert response.status_code == 200, response.text

        searches: Final = _searches_for_marker(strict_gateway.upstream, marker, store_id)
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


@pytest.mark.parametrize(
    ("scope", "error_type"),
    (("key", "key_vector_store_access_denied"), ("team", "team_vector_store_access_denied")),
)
def test_rag_query_is_denied_when_key_or_team_allowlist_excludes_store(
    gateway: Gateway, scope: Literal["key", "team"], error_type: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, scope, "vs_some_other_store")
        marker: Final = f"lit5610 rag query denied {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == error_type, response.text
        assert _searches_for_marker(gateway, marker) == ()


@pytest.mark.parametrize("scope", ("key", "team"))
def test_rag_query_searches_configured_store_when_allowlist_includes_it(
    gateway: Gateway, scope: Literal["key", "team"]
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, scope, CONFIG_STORE_ID)
        marker: Final = f"lit5610 rag query allowed {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)
        assert response.status_code == 200, response.text

        searches: Final = _searches_for_marker(gateway, marker)
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_rag_query_without_key_object_permission_can_search_store(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=_json_array(model))
        marker: Final = f"lit5610 rag query no permission {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)
        assert response.status_code == 200, response.text

        searches: Final = _searches_for_marker(gateway, marker)
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


@pytest.mark.parametrize("scope", ("team", "key"))
def test_no_registry_rag_query_denies_unregistered_store_when_allowlist_excludes(
    no_registry_gateways: tuple[Gateway, Gateway], scope: Literal["team", "key"]
) -> None:
    no_registry_gateway, upstream_gateway = no_registry_gateways
    with no_registry_gateway.scenario() as scenario:
        model: Final = scenario.model()
        store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
        key: Final = _key_for_scope(scenario, model, scope, "vs_some_other_store")
        marker: Final = f"lit5610 no registry denied {scope} {uuid.uuid4().hex}"
        error_type: Final = "team_vector_store_access_denied" if scope == "team" else "key_vector_store_access_denied"

        response: Final = _rag_query(no_registry_gateway, model, marker, key, store_id=store_id)
        searches: Final = _searches_for_marker(upstream_gateway, marker, store_id)

        assert response.status_code == 401, f"{response.text}; scripted_upstream_searches={searches!r}"
        assert response.json()["error"]["type"] == error_type, response.text
        assert searches == ()


def test_no_registry_rag_query_allows_team_allowlisted_unregistered_store(
    no_registry_gateways: tuple[Gateway, Gateway],
) -> None:
    no_registry_gateway, upstream_gateway = no_registry_gateways
    with no_registry_gateway.scenario() as scenario:
        model: Final = scenario.model()
        store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
        key: Final = _key_for_scope(scenario, model, "team", store_id)
        marker: Final = f"lit5610 no registry allowed {uuid.uuid4().hex}"

        response: Final = _rag_query(no_registry_gateway, model, marker, key, store_id=store_id)
        assert response.status_code == 200, response.text

        searches: Final = _searches_for_marker(upstream_gateway, marker, store_id)
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_chat_completions_top_level_retrieval_config_uses_team_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, "team", "vs_some_other_store")
        marker: Final = f"lit5610 chat top-level retrieval config denied {uuid.uuid4().hex}"
        body: Final[JsonObject] = {
            "model": model,
            "messages": _json_array({"role": "user", "content": marker}),
            "retrieval_config": {
                "vector_store_id": CONFIG_STORE_ID,
                "custom_llm_provider": "openai",
                "top_k": 1,
            },
        }

        response: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        observations: Final = upstream_observations(gateway)

        assert response.status_code == 401, f"{response.text}; scripted_upstream_observations={observations!r}"
        assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text


def test_rag_query_alias_denies_store_when_team_allowlist_excludes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, "team", "vs_some_other_store")
        marker: Final = f"lit5610 rag query alias denied {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key, path="/rag/query")

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text
        assert _searches_for_marker(gateway, marker) == ()


def _content_delta(chunk: Mapping[str, JsonValue]) -> str:
    choices: Final = chunk["choices"]
    if not isinstance(choices, list) or not choices:
        return ""
    delta: Final = object_value(object_value(choices[0])["delta"])
    return str(delta.get("content", ""))


def _assert_rag_query_response(
    gateway: Gateway,
    path: str,
    body: Mapping[str, JsonValue],
    key: str,
    stream: bool,
    response_id: str,
) -> None:
    response: Final = gateway.request("POST", path, {**body, "stream": stream}, key=key)
    assert response.status_code == 200, response.text
    if not stream:
        parsed: Final = ChatCompletion.model_validate_json(response.content)
        assert parsed.id == response_id, response.text
        assert parsed.choices[0].message.content == "RAG answer", response.text
        assert parsed.usage is not None and (parsed.usage.prompt_tokens, parsed.usage.completion_tokens) == (10, 5), (
            response.text
        )
        return

    assert response.headers["content-type"].startswith("text/event-stream"), response.text
    frames: Final = tuple(frame for frame in response.text.replace("\r\n", "\n").strip().split("\n\n") if frame)
    assert all(frame.startswith("data: ") for frame in frames), response.text
    assert frames[-1] == "data: [DONE]", response.text
    chunks: Final = tuple(JSON_OBJECT.validate_json(frame[6:]) for frame in frames[:-1])
    assert chunks, response.text
    assert tuple(chunk["id"] for chunk in chunks) == (response_id,) * len(chunks), response.text
    assert "".join(_content_delta(chunk) for chunk in chunks) == "RAG answer", response.text


def test_rag_query_streaming_injects_search_context_and_records_spend(gateway: Gateway) -> None:
    store_id: Final = f"vs_rag_contract_{uuid.uuid4().hex}"
    search_key: Final = f"rag-search-{uuid.uuid4().hex}"
    chat_key: Final = f"rag-chat-{uuid.uuid4().hex}"
    questions: Final = tuple(f"what is in the note {uuid.uuid4().hex}" for _ in range(4))
    context_texts: Final = tuple(f"retrieved context for {question}" for question in questions)
    response_ids: Final = (
        f"chatcmpl-rag-v1-nonstream-{uuid.uuid4().hex}",
        f"chatcmpl-rag-v1-stream-{uuid.uuid4().hex}",
        f"chatcmpl-rag-alias-nonstream-{uuid.uuid4().hex}",
        f"chatcmpl-rag-alias-stream-{uuid.uuid4().hex}",
    )
    expected_messages_by_case: Final = tuple(
        _json_array(
            {"role": "system", "content": "Answer using the note."},
            {"role": "user", "content": f"Context:\n\n{context_text}\n\n"},
            {"role": "user", "content": question},
        )
        for question, context_text in zip(questions, context_texts)
    )
    search_body: Final[dict[str, JsonValue]] = {
        "filters": None,
        "max_num_results": 1,
        "ranking_options": None,
        "rewrite_query": None,
    }
    expected_search_calls: Final = iter(zip(questions, context_texts))
    expected_chat_calls: Final = iter(
        (
            (False, response_ids[0], expected_messages_by_case[0]),
            (True, response_ids[1], expected_messages_by_case[1]),
            (False, response_ids[2], expected_messages_by_case[2]),
            (True, response_ids[3], expected_messages_by_case[3]),
        )
    )

    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if path == f"/v1/vector_stores/{store_id}/search":
            assert request.method == "POST"
            assert request.headers["authorization"] == f"Bearer {search_key}"
            search_question, search_context = next(expected_search_calls)
            assert JSON_OBJECT.validate_json(request.body) == {**search_body, "query": search_question}
            return Reply(
                body=json.dumps(
                    {
                        "object": "vector_store.search_results.page",
                        "search_query": search_question,
                        "data": [
                            {
                                "file_id": "file-rag-context",
                                "filename": "context.txt",
                                "score": 0.9,
                                "attributes": {},
                                "content": [{"type": "text", "text": search_context}],
                            }
                        ],
                        "has_more": False,
                        "next_page": None,
                    }
                ).encode()
            )

        assert path == "/v1/chat/completions", request.target
        assert request.method == "POST"
        assert request.headers["authorization"] == f"Bearer {chat_key}"
        body: Final = JSON_OBJECT.validate_json(request.body)
        expected_stream, response_id, expected_messages = next(expected_chat_calls)
        assert body["messages"] == expected_messages, request.body.decode()
        stream: Final = body.get("stream", False)
        assert stream == expected_stream, request.body.decode()
        model_name: Final = str(body["model"])
        if stream:
            chunks: Final = (
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "RAG "}, "finish_reason": None}],
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model_name,
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
            frames: Final = tuple(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks) + (
                b"data: [DONE]\n\n",
            )
            return Reply(chunks=frames, content_type="text/event-stream")
        return Reply(
            body=json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "RAG answer"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with gateway.scenario() as scenario, wire_server(respond) as wire:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=f"{wire.url}/v1",
            api_key=chat_key,
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        registered: Final = gateway.request(
            "POST",
            "/vector_store/new",
            {
                "vector_store_id": store_id,
                "custom_llm_provider": "openai",
                "litellm_params": {"api_base": f"{wire.url}/v1", "api_key": search_key},
            },
        )
        assert registered.status_code == 200, registered.text
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store_id})
        key: Final = _key_for_scope(scenario, model, "key", store_id)
        body: Final[JsonObject] = {
            "model": model,
            "retrieval_config": {
                "vector_store_id": store_id,
                "custom_llm_provider": "openai",
                "top_k": 1,
            },
        }
        request_bodies: Final = tuple(
            {
                **body,
                "messages": _json_array(
                    {"role": "system", "content": "Answer using the note."},
                    {"role": "user", "content": question},
                ),
            }
            for question in questions
        )
        query_cases: Final[tuple[tuple[str, bool, str], ...]] = (
            ("/v1/rag/query", False, response_ids[0]),
            ("/v1/rag/query", True, response_ids[1]),
            ("/rag/query", False, response_ids[2]),
            ("/rag/query", True, response_ids[3]),
        )
        for (path, stream, response_id), request_body in zip(query_cases, request_bodies):
            _assert_rag_query_response(gateway, path, request_body, key, stream, response_id)

        observed: Final = wire.drain()
        assert tuple((request.method, urlsplit(request.target).path) for request in observed) == (
            ("POST", f"/v1/vector_stores/{store_id}/search"),
            ("POST", "/v1/chat/completions"),
            ("POST", f"/v1/vector_stores/{store_id}/search"),
            ("POST", "/v1/chat/completions"),
            ("POST", f"/v1/vector_stores/{store_id}/search"),
            ("POST", "/v1/chat/completions"),
            ("POST", f"/v1/vector_stores/{store_id}/search"),
            ("POST", "/v1/chat/completions"),
        ), observed
        assert tuple(JSON_OBJECT.validate_json(request.body) for request in observed[::2]) == tuple(
            {**search_body, "query": question} for question in questions
        ), observed
        chat_bodies: Final = tuple(JSON_OBJECT.validate_json(request.body) for request in observed[1::2])
        assert tuple(body["messages"] for body in chat_bodies) == expected_messages_by_case, observed
        assert tuple(body["model"] for body in chat_bodies) == ("gpt-4o-mini",) * 4, observed
        assert tuple(body.get("stream", False) for body in chat_bodies) == (False, True, False, True), observed
        assert all(request.headers["authorization"] == f"Bearer {search_key}" for request in observed[::2])
        assert all(request.headers["authorization"] == f"Bearer {chat_key}" for request in observed[1::2])

        key_hash: Final = sha256(key.encode()).hexdigest()
        spend_rows: Final = eventually(
            lambda: read_rows(
                "SELECT request_id, api_key, spend, prompt_tokens, completion_tokens "
                'FROM "LiteLLM_SpendLogs" WHERE request_id IN (%s, %s, %s, %s)',
                response_ids,
            ),
            lambda rows: len(rows) == 4,
            seconds=70,
        )
        assert {
            (row["request_id"], row["api_key"], float(row["spend"]), row["prompt_tokens"], row["completion_tokens"])
            for row in spend_rows
        } == {(response_id, key_hash, 0.02, 10, 5) for response_id in response_ids}


def test_explicit_false_flag_keeps_legacy_vector_store_outcomes(tmp_path: Path) -> None:
    with gateway_from_environment() as upstream_gateway:
        with owned_proxy(
            upstream_gateway,
            tmp_path,
            _openai_environment(upstream_gateway),
            config=_strict_config(tmp_path, deny_by_default=False),
            remove_environment=REMOVE_OPENAI_API_BASE,
        ) as gateway:
            with gateway.scenario() as scenario:
                model: Final = scenario.model()
                allowed_marker: Final = f"flag-false-allowed-{uuid.uuid4().hex}"
                denied_marker: Final = f"flag-false-denied-{uuid.uuid4().hex}"
                no_permission_key: Final = scenario.key(models=_json_array(model))
                excluding_key: Final = scenario.key(
                    models=_json_array(model), object_permission=_permission_for_stores("vs_some_other_store")
                )

                allowed: Final = _rag_query(gateway, model, allowed_marker, no_permission_key)
                denied: Final = _rag_query(gateway, model, denied_marker, excluding_key)

            assert allowed.status_code == 200, allowed.text
            assert len(_searches_for_marker(upstream_gateway, allowed_marker)) == 1
            assert denied.status_code == 401, denied.text
            assert denied.json()["error"]["type"] == "key_vector_store_access_denied"
            assert _searches_for_marker(upstream_gateway, denied_marker) == ()


def _strict_no_registry_config(directory: Path) -> Path:
    config: Final = object_value(yaml.safe_load(_no_registry_config(directory).read_text()))
    general_settings: Final = object_value(config["general_settings"])
    strict: Final[JsonObject] = {
        **config,
        "general_settings": {**general_settings, "vector_store_deny_by_default": True},
    }
    path: Final = directory / "proxy_vector_store_deny_by_default_no_registry.yaml"
    path.write_text(yaml.safe_dump(strict))
    return path


def test_deny_by_default_without_registry_checks_search_route_and_file_search_tools(tmp_path: Path) -> None:
    with gateway_from_environment() as upstream_gateway:
        with owned_proxy(
            upstream_gateway,
            tmp_path,
            _openai_environment(upstream_gateway),
            config=_strict_no_registry_config(tmp_path),
            remove_environment=REMOVE_OPENAI_API_BASE,
        ) as gateway:
            with gateway.scenario() as scenario:
                model: Final = scenario.model()
                store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
                search_marker: Final = f"lit6035 no registry search {uuid.uuid4().hex}"
                responses_marker: Final = f"lit6035 no registry responses {uuid.uuid4().hex}"
                granted_marker: Final = f"lit6035 no registry granted search {uuid.uuid4().hex}"
                ungranted_key: Final = scenario.key(models=_json_array(model))
                granted_key: Final = scenario.key(
                    models=_json_array(model), object_permission=_permission_for_stores(store_id)
                )

                search_denied: Final = gateway.request(
                    "POST", f"/v1/vector_stores/{store_id}/search", {"query": search_marker}, key=ungranted_key
                )
                responses_denied: Final = gateway.request(
                    "POST",
                    "/v1/responses",
                    {
                        "model": model,
                        "input": responses_marker,
                        "tools": _json_array({"type": "file_search", "vector_store_ids": _json_array(store_id)}),
                    },
                    key=ungranted_key,
                )
                search_granted: Final = gateway.request(
                    "POST", f"/v1/vector_stores/{store_id}/search", {"query": granted_marker}, key=granted_key
                )

            observations: Final = upstream_observations(upstream_gateway)
            denied_observations: Final = tuple(
                observation
                for observation in observations
                if search_marker in str(observation["body"]) or responses_marker in str(observation["body"])
            )
            granted_searches: Final = tuple(
                observation
                for observation in observations
                if observation["path"] == f"/vector_stores/{store_id}/search"
                and granted_marker in str(observation["body"])
            )
            assert search_denied.status_code == 401, search_denied.text
            assert search_denied.json()["error"]["type"] == "key_vector_store_access_denied", search_denied.text
            assert responses_denied.status_code == 401, responses_denied.text
            assert responses_denied.json()["error"]["type"] == "key_vector_store_access_denied", responses_denied.text
            assert denied_observations == ()
            assert search_granted.status_code == 200, search_granted.text
            assert len(granted_searches) == 1, granted_searches


def _auth_cache_subscribers() -> int:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        return int(cache.pubsub_numsub(AUTH_CACHE_INVALIDATION_CHANNEL)[0][1])


def test_revoked_user_grant_stops_working_on_another_proxy(strict_gateway: StrictGateway, tmp_path: Path) -> None:
    subscribers_before_peer: Final = _auth_cache_subscribers()
    with (
        owned_proxy(
            strict_gateway.upstream,
            tmp_path,
            strict_gateway.environment,
            config=strict_gateway.config,
            remove_environment=REMOVE_OPENAI_API_BASE,
        ) as peer,
        strict_gateway.gateway.scenario() as scenario,
    ):
        eventually(_auth_cache_subscribers, lambda count: count > subscribers_before_peer)
        model: Final = scenario.model()
        store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
        user: Final = scenario.user(user_role="internal_user", object_permission=_permission_for_stores(store_id))
        token: Final = strict_gateway.jwt(user)

        def peer_status() -> int:
            marker: Final = f"lit6035 revoked user grant {uuid.uuid4().hex}"
            return peer.request(
                "POST", "/v1/rag/query", _rag_query_body(model, marker, store_id), key=token
            ).status_code

        assert eventually(peer_status, lambda status: status == 200, seconds=30, return_last_on_timeout=True) == 200
        strict_gateway.gateway.post("/user/update", {"user_id": user, "object_permission": _permission_for_stores()})

        assert eventually(peer_status, lambda status: status == 401, seconds=10, return_last_on_timeout=True) == 401
