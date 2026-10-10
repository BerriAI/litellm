from __future__ import annotations

import asyncio
import json
import re
import signal
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import jwt
import psutil
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

TEAM_API_KEY: Final = "integration-rag-team-key"
WILDCARD_API_KEY: Final = "integration-rag-wildcard-key"
SHARED_API_KEY: Final = "integration-rag-shared-key"
ENVIRONMENT_API_KEY: Final = "integration-rag-environment-key"
CALLER_API_KEY: Final = "integration-rag-caller-key"
CREDENTIAL_API_KEY: Final = "integration-rag-credential-key"
GEMINI_API_KEY: Final = "integration-rag-gemini-key"
AZURE_API_KEY: Final = "integration-rag-azure-key"
JWT_KEY_ID: Final = "integration-rag-team-provider-jwt"
WILDCARD_MODEL: Final = "openai/gpt-4o-mini"
OPENAI: Final[Mapping[str, JsonValue]] = MappingProxyType({"custom_llm_provider": "openai"})
DOCUMENT: Final = ("document.txt", b"team provider credentials", "text/plain")
FAILING_STORE_NAMES: Final = MappingProxyType({"rag-upstream-401": 401, "rag-upstream-429": 429})
OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
PREFIXED: Final = re.compile(r"^/(team|wildcard|shared|environment|credential|caller|gemini|azure)(/[^?]*)")
STORE_FILES: Final = re.compile(r"^/v1/vector_stores/([^/]+)/files$")
STORE_SEARCH: Final = re.compile(r"^(?:/v1|/openai)/vector_stores/([^/]+)/search$")
GEMINI_UPLOAD: Final = re.compile(r"^/upload/v1beta/fileSearchStores/([^:]+):uploadToFileSearchStore$")

pytestmark: Final = pytest.mark.timeout(OWNED_PROXY_CELL_SECONDS)

Signature = tuple[str, str, str, str | None]


@dataclass(frozen=True, slots=True)
class Call:
    prefix: str
    method: str
    route: str
    authorization: str | None
    google_key: str | None
    azure_key: str | None

    @property
    def signature(self) -> Signature:
        return (self.prefix, self.method, self.route, self.authorization)


@dataclass(frozen=True, slots=True)
class Tenants:
    shared_model: str
    team_a: str
    team_a_model: str
    team_a_key: str
    restricted_key: str
    viewer_key: str
    jwt_member: str
    team_w_key: str
    team_s_key: str
    teamless_key: str
    credential_name: str
    unreachable_credential_name: str


@dataclass(frozen=True, slots=True)
class RagRig:
    gateway: Gateway
    upstream: Wire
    outage: threading.Event
    tenants: Tenants
    signing_key: rsa.RSAPrivateKey

    def jwt(self, subject: str, team_id: str) -> str:
        issued_at: Final = int(time.time())
        return jwt.encode(
            {"sub": subject, "groups": [team_id], "iat": issued_at, "exp": issued_at + 300},
            self.signing_key,
            algorithm="RS256",
            headers={"kid": JWT_KEY_ID},
        )

    def calls(self) -> tuple[Call, ...]:
        return tuple(_call(request) for request in self.upstream.drain())

    def signatures(self) -> tuple[Signature, ...]:
        return tuple(call.signature for call in self.calls())


def _call(request: Request) -> Call:
    matched: Final = PREFIXED.match(request.target)
    assert matched is not None, request.target
    return Call(
        prefix=matched[1],
        method=request.method,
        route=matched[2],
        authorization=request.headers.get("authorization"),
        google_key=request.headers.get("x-goog-api-key"),
        azure_key=request.headers.get("api-key"),
    )


def _json_body(request: Request) -> Mapping[str, JsonValue]:
    if not request.body or not request.headers.get("content-type", "").startswith("application/json"):
        return MappingProxyType({})
    return object_value(json.loads(request.body))


def _json_reply(payload: Mapping[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def _error_reply(status: int) -> Reply:
    return _json_reply(
        {"error": {"message": f"scripted provider {status}", "type": "invalid_request_error", "code": str(status)}},
        status,
    )


def _vector_store_reply(request: Request, outage: threading.Event) -> Reply:
    if outage.is_set():
        return _error_reply(503)
    name: Final = str(_json_body(request).get("name"))
    failing: Final = FAILING_STORE_NAMES.get(name)
    if failing is not None:
        return _error_reply(failing)
    return _json_reply(
        {
            "id": f"vs_{uuid.uuid4().hex}",
            "object": "vector_store",
            "created_at": int(time.time()),
            "name": name,
            "usage_bytes": 0,
            "status": "completed",
            "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
        }
    )


def _file_reply() -> Reply:
    return _json_reply(
        {
            "id": f"file-{uuid.uuid4().hex}",
            "object": "file",
            "bytes": len(DOCUMENT[1]),
            "created_at": int(time.time()),
            "filename": DOCUMENT[0],
            "purpose": "assistants",
            "status": "processed",
        }
    )


def _attach_reply(request: Request, store_id: str) -> Reply:
    return _json_reply(
        {
            "id": str(_json_body(request).get("file_id")),
            "object": "vector_store.file",
            "created_at": int(time.time()),
            "vector_store_id": store_id,
            "status": "completed",
            "last_error": None,
            "usage_bytes": 0,
        }
    )


def _search_reply(request: Request) -> Reply:
    return _json_reply(
        {
            "object": "vector_store.search_results.page",
            "search_query": _json_body(request).get("query", ""),
            "data": [],
            "has_more": False,
            "next_page": None,
        }
    )


def _sse(payload: Mapping[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chunk(identity: str, model: str, delta: Mapping[str, JsonValue], finish_reason: str | None) -> bytes:
    return _sse(
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        }
    )


def _chat_reply(request: Request) -> Reply:
    body: Final = _json_body(request)
    model: Final = str(body.get("model"))
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    if body.get("stream") is True:
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _chunk(identity, model, {"role": "assistant", "content": "integration response"}, None),
                _chunk(identity, model, {}, "stop"),
                b"data: [DONE]\n\n",
            ),
        )
    return _json_reply(
        {
            "id": identity,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "integration response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def _responses_reply(request: Request) -> Reply:
    identity: Final = uuid.uuid4().hex
    return _json_reply(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": str(_json_body(request).get("model")),
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "integration response", "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    )


def _gemini_reply(request: Request, route: str) -> Reply:
    if request.method == "POST" and route == "/v1beta/fileSearchStores":
        return _json_reply({"name": f"fileSearchStores/rag-{uuid.uuid4().hex}"})
    upload: Final = GEMINI_UPLOAD.match(route)
    if request.method == "POST" and upload is not None:
        session_url: Final = f"http://{request.headers['host']}/gemini/upload-session/{upload[1]}"
        return Reply(body=b"{}", headers={"x-goog-upload-url": session_url})
    if request.method == "PUT" and route.startswith("/upload-session/"):
        return _json_reply({"name": f"documents/{uuid.uuid4().hex}"})
    return _error_reply(404)


def _openai_reply(request: Request, route: str, outage: threading.Event) -> Reply:
    if request.method != "POST":
        return _error_reply(404)
    if route == "/v1/vector_stores":
        return _vector_store_reply(request, outage)
    if route == "/v1/files":
        return _file_reply()
    if route == "/v1/chat/completions":
        return _chat_reply(request)
    if route == "/v1/responses":
        return _responses_reply(request)
    attach: Final = STORE_FILES.match(route)
    if attach is not None:
        return _attach_reply(request, attach[1])
    if STORE_SEARCH.match(route) is not None:
        return _search_reply(request)
    return _error_reply(404)


def _responder(outage: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        matched: Final = PREFIXED.match(request.target)
        if matched is None:
            return _error_reply(404)
        if matched[1] == "gemini":
            return _gemini_reply(request, matched[2])
        return _openai_reply(request, matched[2], outage)

    return respond


def _proxy_config(directory: Path) -> Path:
    base: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    general_settings: Final = object_value(base["general_settings"])
    config: Final = {
        **base,
        "general_settings": {
            **general_settings,
            "enable_jwt_auth": True,
            "litellm_jwtauth": {"user_id_jwt_field": "sub", "team_ids_jwt_field": "groups"},
        },
        "files_settings": [{"custom_llm_provider": "openai"}],
    }
    path: Final = directory / "rag_team_provider_credentials.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _provider_environment(upstream: str) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "OPENAI_API_KEY": ENVIRONMENT_API_KEY,
            "OPENAI_API_BASE": f"{upstream}/environment/v1",
            "OPENAI_BASE_URL": f"{upstream}/environment/v1",
            "GEMINI_API_KEY": GEMINI_API_KEY,
            "GEMINI_API_BASE": f"{upstream}/gemini",
            "AZURE_API_KEY": AZURE_API_KEY,
            "AZURE_API_BASE": f"{upstream}/azure",
            "AZURE_API_VERSION": "2025-04-01-preview",
        }
    )


def _deployment(
    gateway: Gateway, scenario: Scenario, *, name: str, model: str, api_key: str, api_base: str, team_id: str | None
) -> None:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, "api_key": api_key, "api_base": api_base},
            "model_info": {} if team_id is None else {"team_id": team_id},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))


def _delete_credential(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"/credentials/{name}")
    assert response.status_code == 200, response.text


def _credential(gateway: Gateway, scenario: Scenario, api_base: str) -> str:
    name: Final = f"integration-rag-credential-{uuid.uuid4().hex}"
    gateway.post(
        "/credentials",
        {
            "credential_name": name,
            "credential_values": {"api_key": CREDENTIAL_API_KEY, "api_base": api_base},
            "credential_info": {},
        },
    )
    scenario.cleanups.callback(_delete_credential, gateway, name)
    return name


def _tenants(gateway: Gateway, scenario: Scenario, upstream: str) -> Tenants:
    shared_model: Final = f"integration-rag-shared-{uuid.uuid4().hex}"
    _deployment(
        gateway,
        scenario,
        name=shared_model,
        model="openai/gpt-4o-mini",
        api_key=SHARED_API_KEY,
        api_base=f"{upstream}/shared/v1",
        team_id=None,
    )
    team_a_model: Final = f"integration-rag-team-{uuid.uuid4().hex}"
    team_a: Final = scenario.team(models=[team_a_model, shared_model])
    _deployment(
        gateway,
        scenario,
        name=team_a_model,
        model="openai/gpt-4o-mini",
        api_key=TEAM_API_KEY,
        api_base=f"{upstream}/team/v1",
        team_id=team_a,
    )
    viewer: Final = scenario.user(user_role="internal_user_viewer")
    gateway.post("/team/member_add", {"team_id": team_a, "member": {"user_id": viewer, "role": "user"}})
    team_w: Final = scenario.team(models=[shared_model])
    _deployment(
        gateway,
        scenario,
        name="openai/*",
        model="openai/*",
        api_key=WILDCARD_API_KEY,
        api_base=f"{upstream}/wildcard/v1",
        team_id=team_w,
    )
    team_s: Final = scenario.team(models=[shared_model])
    return Tenants(
        shared_model=shared_model,
        team_a=team_a,
        team_a_model=team_a_model,
        team_a_key=scenario.key(team_id=team_a),
        restricted_key=scenario.key(team_id=team_a, models=[shared_model]),
        viewer_key=scenario.key(user_id=viewer, team_id=team_a),
        jwt_member=scenario.member(team_a),
        team_w_key=scenario.key(team_id=team_w),
        team_s_key=scenario.key(team_id=team_s),
        teamless_key=scenario.key(),
        credential_name=_credential(gateway, scenario, f"{upstream}/credential/v1"),
        unreachable_credential_name=_credential(gateway, scenario, _closed_url()),
    )


@pytest.fixture(scope="module")
def rag_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RagRig]:
    signing_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwks_body: Final = json.dumps(
        {
            "keys": [
                {
                    **json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key())),
                    "kid": JWT_KEY_ID,
                    "use": "sig",
                    "alg": "RS256",
                }
            ]
        }
    ).encode()

    def jwks_reply(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=jwks_body)

    outage: Final = threading.Event()
    directory: Final = tmp_path_factory.mktemp("rag_team_provider_credentials")
    with (
        gateway_from_environment() as admin_gateway,
        admin_gateway.scenario() as scenario,
        wire_server(_responder(outage)) as upstream,
        wire_server(jwks_reply) as jwks,
    ):
        tenants: Final = _tenants(admin_gateway, scenario, upstream.url)
        with owned_proxy(
            admin_gateway,
            directory,
            {**_provider_environment(upstream.url), "JWT_PUBLIC_KEY_URL": jwks.url},
            config=_proxy_config(directory),
            remove_environment=("GOOGLE_API_KEY",),
            workers=2,
        ) as gateway:
            yield RagRig(gateway, upstream, outage, tenants, signing_key)


def _ingest_request(name: str, vector_store: Mapping[str, JsonValue]) -> Mapping[str, str]:
    return MappingProxyType(
        {"request": json.dumps({"ingest_options": {"name": name, "vector_store": dict(vector_store)}})}
    )


def _ingest(
    gateway: Gateway,
    key: str,
    vector_store: Mapping[str, JsonValue] = OPENAI,
    *,
    name: str = "litellm-rag-ingest",
    path: str = "/v1/rag/ingest",
) -> httpx.Response:
    return gateway.request_multipart(path, _ingest_request(name, vector_store), {"file": DOCUMENT}, key=key)


def _query(
    gateway: Gateway,
    key: str,
    model: str,
    retrieval_config: Mapping[str, JsonValue],
    *,
    stream: bool = False,
    path: str = "/v1/rag/query",
) -> httpx.Response:
    return gateway.request(
        "POST",
        path,
        {
            "model": model,
            "messages": [{"role": "user", "content": f"query team vector store {uuid.uuid4().hex}"}],
            "retrieval_config": {"top_k": 1, **retrieval_config},
            **({"stream": True} if stream else {}),
        },
        key=key,
    )


def _ingest_signatures(prefix: str, api_key: str, store_id: str) -> tuple[Signature, ...]:
    bearer: Final = f"Bearer {api_key}"
    return (
        (prefix, "POST", "/v1/vector_stores", bearer),
        (prefix, "POST", "/v1/files", bearer),
        (prefix, "POST", f"/v1/vector_stores/{store_id}/files", bearer),
    )


def _search_signature(prefix: str, api_key: str, store_id: str) -> Signature:
    return (prefix, "POST", f"/v1/vector_stores/{store_id}/search", f"Bearer {api_key}")


def _chat_signature(prefix: str, api_key: str) -> Signature:
    return (prefix, "POST", "/v1/chat/completions", f"Bearer {api_key}")


def _completed_store(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    assert body["status"] == "completed", response.text
    return string_value(body["vector_store_id"])


def _store_rows(store_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT vector_store_id, team_id, litellm_params, vector_store_metadata FROM "LiteLLM_ManagedVectorStoresTable" '
        "WHERE vector_store_id = %s",
        (store_id,),
    )


def _closed_url() -> str:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = reserve.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def test_viewer_ingest_and_team_and_jwt_query_use_team_provider_credentials(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants
    store_id: Final = _completed_store(_ingest(rag_rig.gateway, tenants.viewer_key))
    assert rag_rig.signatures() == _ingest_signatures("team", TEAM_API_KEY, store_id)

    team_query: Final = _query(
        rag_rig.gateway, tenants.team_a_key, tenants.team_a_model, {**OPENAI, "vector_store_id": store_id}
    )
    assert team_query.status_code == 200, team_query.text
    assert rag_rig.signatures() == (
        _search_signature("team", TEAM_API_KEY, store_id),
        _chat_signature("team", TEAM_API_KEY),
    )

    jwt_query: Final = _query(
        rag_rig.gateway,
        rag_rig.jwt(tenants.jwt_member, tenants.team_a),
        tenants.team_a_model,
        {**OPENAI, "vector_store_id": store_id},
    )
    assert jwt_query.status_code == 200, jwt_query.text
    assert rag_rig.signatures() == (
        _search_signature("team", TEAM_API_KEY, store_id),
        _chat_signature("team", TEAM_API_KEY),
    )


@pytest.mark.parametrize(
    ("name", "status"), (("rag-upstream-401", 401), ("rag-upstream-429", 429)), ids=("provider_401", "provider_429")
)
def test_failed_ingest_returns_the_provider_status_without_saving_a_store(
    rag_rig: RagRig, name: str, status: int
) -> None:
    before: Final = _store_rows("")

    response: Final = _ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, name=name)

    assert response.status_code == status, response.text
    assert f"scripted provider {status}" in response.text, response.text
    assert rag_rig.signatures() == (("team", "POST", "/v1/vector_stores", f"Bearer {TEAM_API_KEY}"),)
    assert _store_rows("") == before


def test_unreachable_credential_api_base_fails_ingest_without_saving_a_store(rag_rig: RagRig) -> None:
    before: Final = _store_rows("")

    response: Final = _ingest(
        rag_rig.gateway,
        rag_rig.tenants.team_a_key,
        {**OPENAI, "litellm_credential_name": rag_rig.tenants.unreachable_credential_name},
    )

    assert response.status_code == 500, response.text
    assert rag_rig.calls() == ()
    assert _store_rows("") == before


def test_rag_alias_ingest_and_query_use_the_team_wildcard_deployment(rag_rig: RagRig) -> None:
    store_id: Final = _completed_store(_ingest(rag_rig.gateway, rag_rig.tenants.team_w_key, path="/rag/ingest"))
    assert rag_rig.signatures() == _ingest_signatures("wildcard", WILDCARD_API_KEY, store_id)
    rows: Final = _store_rows(store_id)
    assert len(rows) == 1, rows
    assert WILDCARD_API_KEY not in json.dumps(rows), rows

    response: Final = _query(
        rag_rig.gateway,
        rag_rig.tenants.team_w_key,
        WILDCARD_MODEL,
        {**OPENAI, "vector_store_id": store_id},
        path="/rag/query",
    )

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        _search_signature("wildcard", WILDCARD_API_KEY, store_id),
        _chat_signature("wildcard", WILDCARD_API_KEY),
    )


def test_jwt_team_member_ingest_uses_team_provider_credentials(rag_rig: RagRig) -> None:
    token: Final = rag_rig.jwt(rag_rig.tenants.jwt_member, rag_rig.tenants.team_a)

    store_id: Final = _completed_store(_ingest(rag_rig.gateway, token))

    assert rag_rig.signatures() == _ingest_signatures("team", TEAM_API_KEY, store_id)
    rows: Final = _store_rows(store_id)
    assert [row["team_id"] for row in rows] == [rag_rig.tenants.team_a], rows


def test_caller_credential_name_wins_over_the_team_deployment(rag_rig: RagRig) -> None:
    store_id: Final = _completed_store(
        _ingest(
            rag_rig.gateway,
            rag_rig.tenants.team_a_key,
            {**OPENAI, "litellm_credential_name": rag_rig.tenants.credential_name},
        )
    )

    assert rag_rig.signatures() == _ingest_signatures("credential", CREDENTIAL_API_KEY, store_id)


@pytest.mark.parametrize("field", ("api_key", "api_base"))
@pytest.mark.parametrize("value", ("set", "null", "empty"))
def test_caller_api_key_or_api_base_on_ingest_is_rejected_before_any_upstream_call(
    rag_rig: RagRig, field: Literal["api_key", "api_base"], value: Literal["set", "null", "empty"]
) -> None:
    set_values: Final = MappingProxyType({"api_key": CALLER_API_KEY, "api_base": f"{rag_rig.upstream.url}/caller/v1"})
    sent: Final[JsonValue] = MappingProxyType({"set": set_values[field], "null": None, "empty": ""})[value]

    response: Final = _ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, {**OPENAI, field: sent})

    assert response.status_code == 400, response.text
    assert f"'{field}' cannot be set in ingest_options.vector_store" in response.text, response.text
    assert rag_rig.calls() == ()


@pytest.mark.parametrize("blank", (None, ""), ids=("null", "empty"))
def test_blank_caller_credential_name_falls_back_to_the_team_deployment(rag_rig: RagRig, blank: str | None) -> None:
    tenants: Final = rag_rig.tenants

    store_id: Final = _completed_store(
        _ingest(rag_rig.gateway, tenants.team_a_key, {**OPENAI, "litellm_credential_name": blank})
    )
    assert rag_rig.signatures() == _ingest_signatures("team", TEAM_API_KEY, store_id)

    response: Final = _query(
        rag_rig.gateway, tenants.team_a_key, tenants.team_a_model, {**OPENAI, "vector_store_id": store_id}
    )
    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        _search_signature("team", TEAM_API_KEY, store_id),
        _chat_signature("team", TEAM_API_KEY),
    )


@pytest.mark.parametrize("caller", ("teamless_key", "team_s_key", "restricted_key"))
def test_ingest_without_a_usable_team_deployment_uses_the_environment_credentials(
    rag_rig: RagRig, caller: Literal["teamless_key", "team_s_key", "restricted_key"]
) -> None:
    key: Final = getattr(rag_rig.tenants, caller)

    store_id: Final = _completed_store(_ingest(rag_rig.gateway, key))

    assert rag_rig.signatures() == _ingest_signatures("environment", ENVIRONMENT_API_KEY, store_id)


@pytest.mark.parametrize(
    ("caller", "prefix", "api_key"),
    (("team_a_key", "team", TEAM_API_KEY), ("team_s_key", "shared", SHARED_API_KEY)),
    ids=("team_deployment", "shared_deployment"),
)
def test_files_upload_keeps_the_accessible_deployment_fallback(
    rag_rig: RagRig, caller: Literal["team_a_key", "team_s_key"], prefix: str, api_key: str
) -> None:
    response: Final = rag_rig.gateway.request_multipart(
        "/v1/files", {"purpose": "assistants"}, {"file": DOCUMENT}, key=getattr(rag_rig.tenants, caller)
    )

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == ((prefix, "POST", "/v1/files", f"Bearer {api_key}"),)


def test_gemini_ingest_never_receives_the_team_openai_key(rag_rig: RagRig) -> None:
    store_id: Final = _completed_store(
        _ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, {"custom_llm_provider": "gemini"})
    )

    store_name: Final = store_id.removeprefix("fileSearchStores/")
    assert tuple(
        (call.prefix, call.method, call.route, call.google_key, call.authorization) for call in rag_rig.calls()
    ) == (
        ("gemini", "POST", "/v1beta/fileSearchStores", GEMINI_API_KEY, None),
        (
            "gemini",
            "POST",
            f"/upload/v1beta/fileSearchStores/{store_name}:uploadToFileSearchStore",
            GEMINI_API_KEY,
            None,
        ),
        ("gemini", "PUT", f"/upload-session/{store_name}", None, None),
    )


@pytest.mark.parametrize("provider", (7, ["openai"], "", "x" * 5120), ids=("int", "list", "empty", "five_kb"))
def test_malformed_ingest_provider_is_rejected_before_any_upstream_call(rag_rig: RagRig, provider: JsonValue) -> None:
    response: Final = _ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, {"custom_llm_provider": provider})

    assert response.status_code == 400, response.text
    assert rag_rig.calls() == ()


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_unmanaged_store_query_searches_with_the_team_deployment(rag_rig: RagRig, stream: bool) -> None:
    tenants: Final = rag_rig.tenants
    store_id: Final = f"vs_unmanaged_{uuid.uuid4().hex}"

    response: Final = _query(
        rag_rig.gateway,
        tenants.team_a_key,
        tenants.team_a_model,
        {**OPENAI, "vector_store_id": store_id},
        stream=stream,
    )

    assert response.status_code == 200, response.text
    assert "integration response" in response.text, response.text
    assert rag_rig.signatures() == (
        _search_signature("team", TEAM_API_KEY, store_id),
        _chat_signature("team", TEAM_API_KEY),
    )


@pytest.mark.parametrize("case", ("api_key", "api_base", "null_api_key", "empty_api_key"))
def test_caller_credentials_in_retrieval_config_never_reach_an_unmanaged_search(
    rag_rig: RagRig, case: Literal["api_key", "api_base", "null_api_key", "empty_api_key"]
) -> None:
    tenants: Final = rag_rig.tenants
    store_id: Final = f"vs_unmanaged_{uuid.uuid4().hex}"
    caller_fields: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {
            "api_key": {"api_key": CALLER_API_KEY},
            "api_base": {"api_base": f"{rag_rig.upstream.url}/caller/v1"},
            "null_api_key": {"api_key": None},
            "empty_api_key": {"api_key": ""},
        }[case]
    )

    response: Final = _query(
        rag_rig.gateway,
        tenants.team_a_key,
        tenants.team_a_model,
        {**OPENAI, **caller_fields, "vector_store_id": store_id},
    )

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        _search_signature("team", TEAM_API_KEY, store_id),
        _chat_signature("team", TEAM_API_KEY),
    )


def _managed_store(rig: RagRig, scenario: Scenario, litellm_params: Mapping[str, JsonValue]) -> str:
    store_id: Final = f"vs_managed_{uuid.uuid4().hex}"
    rig.gateway.post(
        "/vector_store/new",
        {"vector_store_id": store_id, "custom_llm_provider": "openai", "litellm_params": dict(litellm_params)},
    )
    scenario.cleanups.callback(rig.gateway.post, "/vector_store/delete", {"vector_store_id": store_id})
    return store_id


@pytest.mark.parametrize(
    ("stored", "prefix", "api_key"),
    (("own_key", "caller", CALLER_API_KEY), ("null_key", "team", TEAM_API_KEY), ("empty_key", "team", TEAM_API_KEY)),
    ids=("own_key", "null_key", "empty_key"),
)
def test_managed_store_credentials_decide_the_search_key(
    rag_rig: RagRig, stored: Literal["own_key", "null_key", "empty_key"], prefix: str, api_key: str
) -> None:
    tenants: Final = rag_rig.tenants
    litellm_params: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {
            "own_key": {"api_key": CALLER_API_KEY, "api_base": f"{rag_rig.upstream.url}/caller/v1"},
            "null_key": {"api_key": None},
            "empty_key": {"api_key": ""},
        }[stored]
    )
    with rag_rig.gateway.scenario() as scenario:
        store_id: Final = _managed_store(rag_rig, scenario, litellm_params)

        response: Final = _query(
            rag_rig.gateway, tenants.team_a_key, tenants.team_a_model, {**OPENAI, "vector_store_id": store_id}
        )

        assert response.status_code == 200, response.text
        assert rag_rig.signatures() == (
            _search_signature(prefix, api_key, store_id),
            _chat_signature("team", TEAM_API_KEY),
        )


def test_credential_name_in_retrieval_config_is_rejected_before_search(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = _query(
        rag_rig.gateway,
        tenants.team_a_key,
        tenants.team_a_model,
        {
            **OPENAI,
            "vector_store_id": f"vs_unmanaged_{uuid.uuid4().hex}",
            "litellm_credential_name": tenants.credential_name,
        },
    )

    assert response.status_code == 400, response.text
    assert rag_rig.calls() == ()


def test_azure_query_never_receives_the_team_openai_key(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants
    store_id: Final = f"vs_unmanaged_{uuid.uuid4().hex}"

    response: Final = _query(
        rag_rig.gateway,
        tenants.team_a_key,
        tenants.team_a_model,
        {"custom_llm_provider": "azure", "vector_store_id": store_id},
    )

    assert response.status_code == 200, response.text
    assert tuple((call.prefix, call.route, call.azure_key, call.authorization) for call in rag_rig.calls()) == (
        ("azure", f"/openai/vector_stores/{store_id}/search", AZURE_API_KEY, None),
        ("team", "/v1/chat/completions", None, f"Bearer {TEAM_API_KEY}"),
    )


def test_another_team_cannot_query_a_team_store(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants
    store_id: Final = _completed_store(_ingest(rag_rig.gateway, tenants.team_a_key))
    assert rag_rig.signatures() == _ingest_signatures("team", TEAM_API_KEY, store_id)

    response: Final = _query(
        rag_rig.gateway, tenants.team_w_key, WILDCARD_MODEL, {**OPENAI, "vector_store_id": store_id}
    )

    assert response.status_code == 403, response.text
    assert rag_rig.calls() == ()


@pytest.mark.parametrize("vector_store_id", (7, ["vs_a"], None), ids=("int", "list", "null"))
def test_non_string_vector_store_id_is_rejected_before_search(rag_rig: RagRig, vector_store_id: JsonValue) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = _query(
        rag_rig.gateway, tenants.team_a_key, tenants.team_a_model, {**OPENAI, "vector_store_id": vector_store_id}
    )

    assert response.status_code == 400, response.text
    assert "retrieval_config must contain a string 'vector_store_id'" in response.text, response.text
    assert rag_rig.calls() == ()


def test_empty_vector_store_id_is_rejected_before_search(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = _query(
        rag_rig.gateway, tenants.team_a_key, tenants.team_a_model, {**OPENAI, "vector_store_id": ""}
    )

    assert (response.status_code, rag_rig.signatures()) == (400, ()), response.text


def test_five_kb_vector_store_id_searches_with_the_team_deployment(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants
    store_id: Final = f"vs_{'v' * 5120}"

    response: Final = _query(
        rag_rig.gateway, tenants.team_a_key, tenants.team_a_model, {**OPENAI, "vector_store_id": store_id}
    )

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        _search_signature("team", TEAM_API_KEY, store_id),
        _chat_signature("team", TEAM_API_KEY),
    )


@pytest.mark.parametrize("provider", (7, ["openai"], "", "x" * 5120), ids=("int", "list", "empty", "five_kb"))
def test_malformed_query_provider_fails_without_any_upstream_call(rag_rig: RagRig, provider: JsonValue) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = _query(
        rag_rig.gateway,
        tenants.team_a_key,
        tenants.team_a_model,
        {"custom_llm_provider": provider, "vector_store_id": f"vs_unmanaged_{uuid.uuid4().hex}"},
    )

    assert (response.status_code, rag_rig.signatures()) == (500, ()), response.text


def test_jwt_route_outside_the_team_allowed_routes_names_the_route(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = rag_rig.gateway.request(
        "POST",
        "/v1beta/interactions",
        {"model": tenants.team_a_model, "input": "hello"},
        key=rag_rig.jwt(tenants.jwt_member, tenants.team_a),
    )

    assert response.status_code == 403, response.text
    assert (
        f"Team {tenants.team_a} can access model {tenants.team_a_model} but route /v1beta/interactions "
        "is not in litellm_jwtauth.team_allowed_routes"
    ) in response.text, response.text
    assert rag_rig.calls() == ()


def _llm_body(route: str, model: str) -> dict[str, JsonValue]:
    prompt: Final = f"hello {uuid.uuid4().hex}"
    if route == "/v1/responses":
        return {"model": model, "input": prompt}
    if route == "/v1/messages":
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]}
    return {"model": model, "messages": [{"role": "user", "content": prompt}]}


@pytest.mark.parametrize("route", ("/v1/chat/completions", "/v1/messages", "/v1/responses"))
def test_jwt_team_member_keeps_its_llm_routes(rag_rig: RagRig, route: str) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = rag_rig.gateway.request(
        "POST",
        route,
        _llm_body(route, tenants.team_a_model),
        key=rag_rig.jwt(tenants.jwt_member, tenants.team_a),
        headers={"anthropic-version": "2023-06-01"},
    )

    assert response.status_code == 200, response.text
    calls: Final = rag_rig.calls()
    assert [(call.prefix, call.authorization) for call in calls] == [("team", f"Bearer {TEAM_API_KEY}")], calls


async def _ingest_burst(rig: RagRig, names: tuple[str, ...]) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(base_url=str(rig.gateway.client.base_url), timeout=60, trust_env=False) as client:
        return tuple(
            await asyncio.gather(
                *(
                    client.post(
                        "/v1/rag/ingest",
                        data=_ingest_request(name, OPENAI),
                        files={"file": DOCUMENT},
                        headers={"Authorization": f"Bearer {rig.tenants.team_a_key}"},
                    )
                    for name in names
                )
            )
        )


def _burst_names() -> tuple[str, ...]:
    return tuple(f"rag-burst-{uuid.uuid4().hex}" for _ in range(10))


def _all_ingest_signatures(store_ids: Iterable[str]) -> list[Signature]:
    return sorted(chain.from_iterable(_ingest_signatures("team", TEAM_API_KEY, store_id) for store_id in store_ids))


async def test_ingest_burst_during_a_provider_outage_fails_each_request_then_recovers(rag_rig: RagRig) -> None:
    before: Final = _store_rows("")
    rag_rig.outage.set()
    try:
        failed: Final = await _ingest_burst(rag_rig, _burst_names())
    finally:
        rag_rig.outage.clear()

    assert [response.status_code for response in failed] == [503] * 10, [response.text for response in failed]
    outage_calls: Final = rag_rig.signatures()
    assert outage_calls == (("team", "POST", "/v1/vector_stores", f"Bearer {TEAM_API_KEY}"),) * 10, outage_calls
    assert _store_rows("") == before

    recovered: Final = await _ingest_burst(rag_rig, _burst_names())

    store_ids: Final = tuple(_completed_store(response) for response in recovered)
    assert len(frozenset(store_ids)) == 10, store_ids
    assert sorted(rag_rig.signatures()) == _all_ingest_signatures(store_ids)
    assert [[row["team_id"] for row in _store_rows(store_id)] for store_id in store_ids] == [
        [rag_rig.tenants.team_a]
    ] * 10


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


async def _query_burst(gateway: Gateway, key: str, model: str, count: int) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(
                client.post(
                    "/v1/rag/query",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": f"held query {uuid.uuid4().hex}"}],
                        "retrieval_config": {**OPENAI, "vector_store_id": f"vs_burst_{uuid.uuid4().hex}", "top_k": 1},
                    },
                    headers={"Authorization": f"Bearer {key}"},
                )
                for _ in range(count)
            ),
            return_exceptions=True,
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, httpx.Response))


async def test_worker_sigkill_mid_query_burst_leaves_the_sibling_on_team_credentials(tmp_path: Path) -> None:
    release: Final = threading.Event()
    held_searches: Final[SimpleQueue[str]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        matched: Final = PREFIXED.match(request.target)
        if matched is None:
            return _error_reply(404)
        if STORE_SEARCH.match(matched[2]) is not None:
            held_searches.put(request.target)
            assert release.wait(timeout=60), "The burst was never released"
            return _search_reply(request)
        return _openai_reply(request, matched[2], threading.Event())

    with (
        gateway_from_environment() as admin_gateway,
        admin_gateway.scenario() as scenario,
        wire_server(respond) as upstream,
    ):
        model: Final = f"integration-rag-kill-{uuid.uuid4().hex}"
        team_id: Final = scenario.team(models=[model])
        _deployment(
            admin_gateway,
            scenario,
            name=model,
            model="openai/gpt-4o-mini",
            api_key=TEAM_API_KEY,
            api_base=f"{upstream.url}/team/v1",
            team_id=team_id,
        )
        key: Final = scenario.key(team_id=team_id)
        with owned_proxy_process(
            admin_gateway,
            tmp_path,
            _provider_environment(upstream.url),
            remove_environment=("GOOGLE_API_KEY",),
            workers=2,
        ) as owned:
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(_query_burst(owned.gateway, key, model, 20))
            await asyncio.to_thread(eventually, held_searches.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, upstream.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert [response.status_code for response in served] == [200] * held_by[survivor_pid], held_by
            (follow_up,) = await _query_burst(owned.gateway, key, model, 1)
            assert follow_up.status_code == 200, follow_up.text
            calls: Final = tuple(_call(request) for request in upstream.drain())
            assert {(call.prefix, call.authorization) for call in calls} == {("team", f"Bearer {TEAM_API_KEY}")}, calls
            assert sum(1 for call in calls if call.route == "/v1/chat/completions") == held_by[survivor_pid] + 1, calls
