from __future__ import annotations

import asyncio
import json
import re
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

import anthropic
import httpx
import jwt
import openai
import psutil
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import (
    JSON_OBJECT,
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
FORWARDED_API_KEY: Final = "integration-rag-forwarded-key"
CREDENTIAL_API_KEY: Final = "integration-rag-credential-key"
GEMINI_API_KEY: Final = "integration-rag-gemini-key"
AZURE_API_KEY: Final = "integration-rag-azure-key"
JWT_KEY_ID: Final = "integration-rag-team-provider-jwt"
WILDCARD_MODEL: Final = "openai/gpt-4o-mini"
BARE_WILDCARD_MODEL: Final = "gpt-4o-mini"
OPENAI: Final[Mapping[str, JsonValue]] = MappingProxyType({"custom_llm_provider": "openai"})
DOCUMENT: Final = ("document.txt", b"team provider credentials", "text/plain")
BATCH_INPUT_ROW: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "custom_id": "integration-rag-batch",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": BARE_WILDCARD_MODEL, "messages": [{"role": "user", "content": "batch input"}]},
    }
)
BATCH_INPUT: Final = ("batch.jsonl", (json.dumps(dict(BATCH_INPUT_ROW)) + "\n").encode(), "application/jsonl")
FAILING_STORE_NAMES: Final = MappingProxyType({"rag-upstream-401": 401, "rag-upstream-429": 429})
OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
PREFIXED: Final = re.compile(r"^/(team|wildcard|shared|environment|credential|caller|gemini|azure|bedrock)(/[^?]*)")
STORE_FILES: Final = re.compile(r"^/v1/vector_stores/([^/]+)/files$")
FILE_CONTENT: Final = re.compile(r"^/v1/files/([^/]+)/content$")
FILE_ITEM: Final = re.compile(r"^/v1/files/([^/]+)$")
BATCH_ITEM: Final = re.compile(r"^/v1/batches/([^/]+)$")
BATCH_CANCEL: Final = re.compile(r"^/v1/batches/([^/]+)/cancel$")
BEDROCK_DATA_SOURCES: Final = re.compile(r"^/knowledgebases/([^/]+)/datasources/$")
BEDROCK_DATA_SOURCE: Final = re.compile(r"^/knowledgebases/([^/]+)/datasources/([^/]+)$")
EMPTY_KNOWLEDGE_BASE: Final = "EMPTYKB000"
BUCKETLESS_KNOWLEDGE_BASE: Final = "NOBUCKET00"
WEB_DATA_SOURCE: Final = "WEBSOURCE0"
STORE_SEARCH: Final = re.compile(r"^(?:/v1|/openai)/vector_stores/([^/]+)/search$")
GEMINI_UPLOAD: Final = re.compile(r"^/upload/v1beta/fileSearchStores/([^:]+):uploadToFileSearchStore$")

JWT_TEAM_COUNT: Final = 10
AWS_AMBIENT_CREDENTIALS: Final = (
    "AWS_PROFILE",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ROLE_ARN",
    "AWS_ROLE_NAME",
    "AWS_SESSION_NAME",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
)

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
    global_model: str
    team_a: str
    team_a_model: str
    team_a_key: str
    restricted_key: str
    viewer_key: str
    jwt_member: str
    team_w: str
    team_w_key: str
    wildcard_bare_key: str
    wildcard_prefixed_key: str
    wildcard_global_key: str
    team_s: str
    team_s_key: str
    teamless_key: str
    jwt_teams: tuple[str, ...]
    jwt_team_model: str
    jwt_multi_member: str
    azure_restricted_key: str
    credential_name: str
    unreachable_credential_name: str


@dataclass(frozen=True, slots=True)
class RagRig:
    gateway: Gateway
    upstream: Wire
    outage: threading.Event
    tenants: Tenants
    signing_key: rsa.RSAPrivateKey
    scenario: Scenario

    def jwt(self, subject: str, *team_ids: JsonValue) -> str:
        issued_at: Final = int(time.time())
        return jwt.encode(
            {"sub": subject, "groups": list(team_ids), "iat": issued_at, "exp": issued_at + 300},
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


def _owned_by(prefix: str, identity: str) -> bool:
    return identity.startswith((f"file-{prefix}-", f"batch_{prefix}_"))


def _file_object(identity: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "file",
        "bytes": len(DOCUMENT[1]),
        "created_at": int(time.time()),
        "filename": DOCUMENT[0],
        "purpose": "assistants",
        "status": "processed",
    }


def _file_reply(prefix: str) -> Reply:
    return _json_reply(_file_object(f"file-{prefix}-{uuid.uuid4().hex}"))


def _batch_input_reply() -> Reply:
    return Reply(status=200, body=BATCH_INPUT[1])


def _batch_object(identity: str, input_file_id: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": input_file_id,
        "completion_window": "24h",
        "status": status,
        "created_at": int(time.time()),
        "errors": None,
        "output_file_id": None,
        "error_file_id": None,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
        "metadata": None,
    }


def _batch_reply(request: Request, prefix: str) -> Reply:
    body: Final = _json_body(request)
    return _json_reply(
        _batch_object(f"batch_{prefix}_{uuid.uuid4().hex}", str(body.get("input_file_id")), "validating")
    )


def _list_reply() -> Reply:
    return _json_reply({"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False})


def _owned_reply(prefix: str, identity: str, payload: Mapping[str, JsonValue]) -> Reply:
    return _json_reply(payload) if _owned_by(prefix, identity) else _error_reply(404)


def _openai_get(route: str, prefix: str) -> Reply:
    if route in {"/v1/files", "/v1/batches"}:
        return _list_reply()
    content: Final = FILE_CONTENT.match(route)
    if content is not None:
        return _batch_input_reply() if _owned_by(prefix, content[1]) else _error_reply(404)
    file_item: Final = FILE_ITEM.match(route)
    if file_item is not None:
        return _owned_reply(prefix, file_item[1], _file_object(file_item[1]))
    batch_item: Final = BATCH_ITEM.match(route)
    if batch_item is not None:
        return _owned_reply(prefix, batch_item[1], _batch_object(batch_item[1], f"file-{prefix}-input", "validating"))
    return _error_reply(404)


def _openai_delete(route: str, prefix: str) -> Reply:
    file_item: Final = FILE_ITEM.match(route)
    if file_item is None:
        return _error_reply(404)
    return _owned_reply(prefix, file_item[1], {"id": file_item[1], "object": "file", "deleted": True})


def _embedding_reply(request: Request) -> Reply:
    body: Final = _json_body(request)
    inputs: Final = body.get("input")
    count: Final = len(inputs) if isinstance(inputs, list) else 1
    return _json_reply(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": index, "embedding": [0.1, 0.2, 0.3]} for index in range(count)],
            "model": str(body.get("model")),
            "usage": {"prompt_tokens": count, "total_tokens": count},
        }
    )


def _bedrock_reply(request: Request, route: str) -> Reply:
    listed: Final = BEDROCK_DATA_SOURCES.match(route)
    if request.method == "POST" and listed is not None:
        summary: Final[dict[str, JsonValue]] = {
            "knowledgeBaseId": listed[1],
            "dataSourceId": WEB_DATA_SOURCE,
            "name": "integration-web-source",
            "status": "AVAILABLE",
            "updatedAt": "2026-01-01T00:00:00Z",
        }
        return _json_reply({"dataSourceSummaries": [] if listed[1] == EMPTY_KNOWLEDGE_BASE else [summary]})
    detail: Final = BEDROCK_DATA_SOURCE.match(route)
    if request.method == "GET" and detail is not None:
        return _json_reply(
            {
                "dataSource": {
                    "knowledgeBaseId": detail[1],
                    "dataSourceId": detail[2],
                    "name": "integration-web-source",
                    "status": "AVAILABLE",
                    "dataSourceConfiguration": {"type": "WEB"},
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z",
                }
            }
        )
    return _error_reply(404)


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
    completed: Final = _responses_object(identity, str(_json_body(request).get("model")))
    if _json_body(request).get("stream") is not True:
        return _json_reply(completed)
    events: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**completed, "status": "in_progress", "output": []},
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": f"msg_{identity}",
                "output_index": 0,
                "content_index": 0,
                "delta": "integration response",
                "logprobs": [],
            },
        ),
        ("response.completed", {"type": "response.completed", "sequence_number": 2, "response": completed}),
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events),
    )


def _responses_object(identity: str, model: str) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
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


def _openai_reply(request: Request, prefix: str, route: str, outage: threading.Event) -> Reply:
    if request.method == "GET":
        return _openai_get(route, prefix)
    if request.method == "DELETE":
        return _openai_delete(route, prefix)
    if request.method != "POST":
        return _error_reply(404)
    if route == "/v1/vector_stores":
        return _vector_store_reply(request, outage)
    if route in {"/v1/files", "/openai/files"}:
        return _file_reply(prefix)
    if route == "/v1/batches":
        return _batch_reply(request, prefix)
    cancelled: Final = BATCH_CANCEL.match(route)
    if cancelled is not None:
        return _owned_reply(prefix, cancelled[1], _batch_object(cancelled[1], f"file-{prefix}-input", "cancelling"))
    if route == "/v1/chat/completions":
        return _chat_reply(request)
    if route == "/v1/responses":
        return _responses_reply(request)
    if route == "/v1/embeddings":
        return _embedding_reply(request)
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
        if matched[1] == "bedrock":
            return _bedrock_reply(request, matched[2])
        return _openai_reply(request, matched[1], matched[2], outage)

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
            "forward_llm_provider_auth_headers": True,
        },
        "files_settings": [{"custom_llm_provider": "openai"}, {"custom_llm_provider": "azure"}],
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


def _aws_environment(upstream: str) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "AWS_ENDPOINT_URL_BEDROCK_AGENT": f"{upstream}/bedrock",
            "AWS_ACCESS_KEY_ID": "integration-rag-aws-access-key",
            "AWS_SECRET_ACCESS_KEY": "integration-rag-aws-secret-key",
            "AWS_REGION_NAME": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "true",
        }
    )


def _jwt_team_key(team_id: str) -> str:
    return f"integration-rag-jwt-{team_id}"


def _jwt_team(gateway: Gateway, scenario: Scenario, upstream: str, model: str, member: str) -> str:
    team_id: Final = scenario.team(models=[model])
    _deployment(
        gateway,
        scenario,
        name=model,
        model="openai/gpt-4o-mini",
        api_key=_jwt_team_key(team_id),
        api_base=f"{upstream}/team/v1",
        team_id=team_id,
    )
    gateway.post("/team/member_add", {"team_id": team_id, "member": {"user_id": member, "role": "user"}})
    return team_id


def _deployment(
    gateway: Gateway,
    scenario: Scenario,
    *,
    name: str,
    model: str,
    api_key: str,
    api_base: str,
    team_id: str | None,
    model_id: str | None = None,
) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, "api_key": api_key, "api_base": api_base},
            "model_info": ({} if team_id is None else {"team_id": team_id})
            | ({} if model_id is None else {"id": model_id}),
        },
    )
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return model_id


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
    global_model: Final = f"openai/integration-rag-global-{uuid.uuid4().hex}"
    _deployment(
        gateway,
        scenario,
        name=global_model,
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
    team_z: Final = scenario.team(models=[shared_model])
    _deployment(
        gateway,
        scenario,
        name="azure/*",
        model="azure/*",
        api_key=WILDCARD_API_KEY,
        api_base=f"{upstream}/azure",
        team_id=team_z,
    )
    jwt_team_model: Final = f"integration-rag-jwt-{uuid.uuid4().hex}"
    jwt_multi_member: Final = scenario.user(user_role="internal_user")
    return Tenants(
        shared_model=shared_model,
        global_model=global_model,
        team_a=team_a,
        team_a_model=team_a_model,
        team_a_key=scenario.key(team_id=team_a),
        restricted_key=scenario.key(team_id=team_a, models=[shared_model]),
        viewer_key=scenario.key(user_id=viewer, team_id=team_a),
        jwt_member=scenario.member(team_a),
        team_w=team_w,
        team_w_key=scenario.key(team_id=team_w),
        wildcard_bare_key=scenario.key(team_id=team_w, models=[BARE_WILDCARD_MODEL]),
        wildcard_prefixed_key=scenario.key(team_id=team_w, models=[WILDCARD_MODEL]),
        wildcard_global_key=scenario.key(team_id=team_w, models=[global_model]),
        team_s=team_s,
        team_s_key=scenario.key(team_id=team_s),
        teamless_key=scenario.key(),
        jwt_teams=tuple(
            _jwt_team(gateway, scenario, upstream, jwt_team_model, jwt_multi_member) for _ in range(JWT_TEAM_COUNT)
        ),
        jwt_team_model=jwt_team_model,
        jwt_multi_member=jwt_multi_member,
        azure_restricted_key=scenario.key(team_id=team_z, models=["azure/gpt-4o-mini"]),
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
        with (
            owned_proxy(
                admin_gateway,
                directory,
                {
                    **_provider_environment(upstream.url),
                    **_aws_environment(upstream.url),
                    "JWT_PUBLIC_KEY_URL": jwks.url,
                },
                config=_proxy_config(directory),
                remove_environment=("GOOGLE_API_KEY", *AWS_AMBIENT_CREDENTIALS),
                workers=2,
            ) as gateway,
            gateway.scenario() as owned_scenario,
        ):
            yield RagRig(gateway, upstream, outage, tenants, signing_key, owned_scenario)


@pytest.fixture(autouse=True)
def _drain_calls_an_earlier_failed_cell_left(request: pytest.FixtureRequest) -> None:
    if "rag_rig" in request.fixturenames:
        request.getfixturevalue("rag_rig").upstream.drain()


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


def _completed_store(rig: RagRig, response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    assert body["status"] == "completed", response.text
    store_id: Final = string_value(body["vector_store_id"])
    rig.scenario.cleanups.callback(rig.gateway.post, "/vector_store/delete", {"vector_store_id": store_id})
    return store_id


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
    store_id: Final = _completed_store(rag_rig, _ingest(rag_rig.gateway, tenants.viewer_key))
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
    store_id: Final = _completed_store(
        rag_rig, _ingest(rag_rig.gateway, rag_rig.tenants.team_w_key, path="/rag/ingest")
    )
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

    store_id: Final = _completed_store(rag_rig, _ingest(rag_rig.gateway, token))

    assert rag_rig.signatures() == _ingest_signatures("team", TEAM_API_KEY, store_id)
    rows: Final = _store_rows(store_id)
    assert [row["team_id"] for row in rows] == [rag_rig.tenants.team_a], rows


def test_caller_credential_name_wins_over_the_team_deployment(rag_rig: RagRig) -> None:
    store_id: Final = _completed_store(
        rag_rig,
        _ingest(
            rag_rig.gateway,
            rag_rig.tenants.team_a_key,
            {**OPENAI, "litellm_credential_name": rag_rig.tenants.credential_name},
        ),
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
        rag_rig, _ingest(rag_rig.gateway, tenants.team_a_key, {**OPENAI, "litellm_credential_name": blank})
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

    store_id: Final = _completed_store(rag_rig, _ingest(rag_rig.gateway, key))

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


@pytest.mark.parametrize(
    ("caller", "model"),
    (("wildcard_bare_key", BARE_WILDCARD_MODEL), ("wildcard_prefixed_key", WILDCARD_MODEL)),
    ids=("bare_model", "prefixed_model"),
)
def test_key_restricted_to_a_model_under_the_team_wildcard_uses_the_team_deployment(
    rag_rig: RagRig, caller: Literal["wildcard_bare_key", "wildcard_prefixed_key"], model: str
) -> None:
    key: Final = getattr(rag_rig.tenants, caller)

    store_id: Final = _completed_store(rag_rig, _ingest(rag_rig.gateway, key))
    assert rag_rig.signatures() == _ingest_signatures("wildcard", WILDCARD_API_KEY, store_id)
    rows: Final = _store_rows(store_id)
    assert [row["team_id"] for row in rows] == [rag_rig.tenants.team_w], rows
    assert WILDCARD_API_KEY not in json.dumps(rows), rows

    response: Final = _query(rag_rig.gateway, key, model, {**OPENAI, "vector_store_id": store_id})

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        _search_signature("wildcard", WILDCARD_API_KEY, store_id),
        _chat_signature("wildcard", WILDCARD_API_KEY),
    )


def test_key_restricted_to_a_global_model_under_the_team_wildcard_splits_rag_and_chat(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants

    store_id: Final = _completed_store(rag_rig, _ingest(rag_rig.gateway, tenants.wildcard_global_key))
    assert rag_rig.signatures() == _ingest_signatures("wildcard", WILDCARD_API_KEY, store_id)

    response: Final = _query(
        rag_rig.gateway, tenants.wildcard_global_key, tenants.global_model, {**OPENAI, "vector_store_id": store_id}
    )

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        _search_signature("wildcard", WILDCARD_API_KEY, store_id),
        _chat_signature("shared", SHARED_API_KEY),
    )


def test_files_upload_by_a_key_restricted_under_the_team_wildcard_uses_the_team_deployment(rag_rig: RagRig) -> None:
    response: Final = rag_rig.gateway.request_multipart(
        "/v1/files", {"purpose": "assistants"}, {"file": DOCUMENT}, key=rag_rig.tenants.wildcard_bare_key
    )

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (("wildcard", "POST", "/v1/files", f"Bearer {WILDCARD_API_KEY}"),)


@pytest.mark.parametrize("batch_options", ({}, {"litellm_credential_name": "no-such-credential"}), ids=("plain", "credential_name"))
@pytest.mark.parametrize(
    ("caller", "prefix", "api_key", "reads_input"),
    (("team_a_key", "team", TEAM_API_KEY, False), ("wildcard_bare_key", "wildcard", WILDCARD_API_KEY, True)),
    ids=("team_key", "restricted_wildcard_key"),
)
def test_batch_create_reads_the_input_and_creates_the_batch_through_the_team_deployment(
    rag_rig: RagRig,
    caller: Literal["team_a_key", "wildcard_bare_key"],
    prefix: str,
    api_key: str,
    reads_input: bool,
    batch_options: Mapping[str, JsonValue],
) -> None:
    key: Final = getattr(rag_rig.tenants, caller)
    uploaded: Final = rag_rig.gateway.request_multipart(
        "/v1/files", {"purpose": "batch"}, {"file": BATCH_INPUT}, key=key
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id: Final = string_value(object_value(uploaded.json())["id"])

    created: Final = rag_rig.gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h", **batch_options},
        key=key,
    )

    assert created.status_code == 200, created.text
    assert object_value(created.json())["input_file_id"] == file_id, created.text
    bearer: Final = f"Bearer {api_key}"
    input_read: Final = ((prefix, "GET", f"/v1/files/{file_id}/content", bearer),) if reads_input else ()
    assert rag_rig.signatures() == (
        (prefix, "POST", "/v1/files", bearer),
        *input_read,
        (prefix, "POST", "/v1/batches", bearer),
    )


def test_gemini_ingest_never_receives_the_team_openai_key(rag_rig: RagRig) -> None:
    store_id: Final = _completed_store(
        rag_rig, _ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, {"custom_llm_provider": "gemini"})
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
    store_id: Final = _completed_store(rag_rig, _ingest(rag_rig.gateway, tenants.team_a_key))
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
    assert "vector_store_id must be a non-empty string" in response.text, response.text
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

    store_ids: Final = tuple(_completed_store(rag_rig, response) for response in recovered)
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
        return _openai_reply(request, matched[1], matched[2], threading.Event())

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
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                assert held_by[survivor_pid] >= 10, held_by
                assert [response.status_code for response in served] == [200] * held_by[survivor_pid], held_by
            (follow_up,) = await _query_burst(owned.gateway, key, model, 1)
            assert follow_up.status_code == 200, follow_up.text
            calls: Final = tuple(_call(request) for request in upstream.drain())
            assert {(call.prefix, call.authorization) for call in calls} == {("team", f"Bearer {TEAM_API_KEY}")}, calls
            assert sum(1 for call in calls if call.route == "/v1/chat/completions") == held_by[survivor_pid] + 1, calls


def _ingest_and_drain(
    gateway: Gateway, scenario: Scenario, key: str, upstream: Wire
) -> tuple[str, tuple[Signature, ...]]:
    upstream.drain()
    response: Final = _ingest(gateway, key)
    assert response.status_code == 200, response.text
    store_id: Final = string_value(object_value(response.json())["vector_store_id"])
    scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store_id})
    calls: Final = tuple(_call(request) for request in upstream.drain())
    return store_id, tuple(call.signature for call in calls if call.route != "/v1/models")


def _ingests_after_editing(
    gateway: Gateway, scenario: Scenario, key: str, upstream: Wire, model_id: str
) -> tuple[tuple[str, tuple[Signature, ...]], ...]:
    edited: Final = gateway.request("PATCH", f"/model/{model_id}/update", {"litellm_params": {"timeout": 30}})
    assert edited.status_code == 200, edited.text
    return tuple(_ingest_and_drain(gateway, scenario, key, upstream) for _ in range(4))


def test_a_team_with_two_openai_deployments_keeps_ingesting_through_the_older_one_after_either_is_edited(
    tmp_path: Path,
) -> None:
    with (
        gateway_from_environment() as admin_gateway,
        admin_gateway.scenario() as scenario,
        wire_server(_responder(threading.Event())) as upstream,
    ):
        team_model: Final = f"integration-rag-team-{uuid.uuid4().hex}"
        wildcard_model: Final = f"integration-rag-wildcard-{uuid.uuid4().hex}"
        team_id: Final = scenario.team(models=[team_model, wildcard_model])
        team_deployment: Final = (
            _deployment(
                admin_gateway,
                scenario,
                name=team_model,
                model="openai/gpt-4o-mini",
                api_key=TEAM_API_KEY,
                api_base=f"{upstream.url}/team/v1",
                team_id=team_id,
                model_id=f"z-{uuid.uuid4().hex}",
            ),
            "team",
            TEAM_API_KEY,
        )
        wildcard_deployment: Final = (
            _deployment(
                admin_gateway,
                scenario,
                name=wildcard_model,
                model="openai/gpt-4o-mini",
                api_key=WILDCARD_API_KEY,
                api_base=f"{upstream.url}/wildcard/v1",
                team_id=team_id,
                model_id=f"a-{uuid.uuid4().hex}",
            ),
            "wildcard",
            WILDCARD_API_KEY,
        )
        (picked_id, picked_prefix, picked_key), (other_id, _, _) = team_deployment, wildcard_deployment
        key: Final = scenario.key(team_id=team_id)
        with (
            owned_proxy(
                admin_gateway,
                tmp_path,
                _provider_environment(upstream.url),
                remove_environment=("GOOGLE_API_KEY",),
                workers=2,
            ) as gateway,
            gateway.scenario() as owned_scenario,
        ):
            ingests: Final = (
                _ingest_and_drain(gateway, owned_scenario, key, upstream),
                *_ingests_after_editing(gateway, owned_scenario, key, upstream, picked_id),
                *_ingests_after_editing(gateway, owned_scenario, key, upstream, other_id),
            )

            assert [calls for _, calls in ingests] == [
                _ingest_signatures(picked_prefix, picked_key, store_id) for store_id, _ in ingests
            ]


@pytest.mark.parametrize(
    ("stored", "prefix", "api_key"),
    (("own_key", "caller", CALLER_API_KEY), ("null_key", "team", TEAM_API_KEY)),
    ids=("own_key", "null_key"),
)
def test_ingest_into_a_managed_store_uses_its_own_key_else_the_team_key(
    rag_rig: RagRig, stored: Literal["own_key", "null_key"], prefix: str, api_key: str
) -> None:
    litellm_params: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {
            "own_key": {"api_key": CALLER_API_KEY, "api_base": f"{rag_rig.upstream.url}/caller/v1"},
            "null_key": {"api_key": None},
        }[stored]
    )
    with rag_rig.gateway.scenario() as scenario:
        store_id: Final = _managed_store(rag_rig, scenario, litellm_params)

        response: Final = _ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, {**OPENAI, "vector_store_id": store_id})

        assert response.status_code == 200, response.text
        assert object_value(response.json())["vector_store_id"] == store_id, response.text
        bearer: Final = f"Bearer {api_key}"
        assert rag_rig.signatures() == (
            (prefix, "POST", "/v1/files", bearer),
            (prefix, "POST", f"/v1/vector_stores/{store_id}/files", bearer),
        )


def _file_id_ingest(gateway: Gateway, key: str, vector_store: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/rag/ingest",
        {
            "file_id": "file-team-existing",
            "ingest_options": {"name": "litellm-rag-ingest", "vector_store": dict(vector_store)},
        },
        key=key,
    )


@pytest.mark.parametrize(
    ("vector_store", "ingestion"),
    (
        ({"custom_llm_provider": "gemini"}, "GeminiRAGIngestion"),
        ({"custom_llm_provider": "bedrock", "vector_store_id": BUCKETLESS_KNOWLEDGE_BASE}, "BedrockRAGIngestion"),
        ({"custom_llm_provider": "s3_vectors", "vector_bucket_name": "integration-bucket"}, "S3VectorsRAGIngestion"),
        ({"custom_llm_provider": "vertex_ai"}, "VertexAIRAGIngestion"),
    ),
    ids=("gemini", "bedrock", "s3_vectors", "vertex_ai"),
)
def test_existing_file_id_on_a_store_that_cannot_take_one_is_rejected_before_any_upstream_call(
    rag_rig: RagRig, vector_store: Mapping[str, JsonValue], ingestion: str
) -> None:
    before: Final = _store_rows("")

    response: Final = _file_id_ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, vector_store)

    assert response.status_code == 400, response.text
    assert f"{ingestion} does not support ingesting an existing file_id" in response.text, response.text
    assert rag_rig.calls() == ()
    assert _store_rows("") == before


def test_openai_file_id_without_a_vector_store_id_is_rejected_before_any_upstream_call(rag_rig: RagRig) -> None:
    before: Final = _store_rows("")

    response: Final = _file_id_ingest(rag_rig.gateway, rag_rig.tenants.team_a_key, OPENAI)

    assert response.status_code == 400, response.text
    assert "vector_store_id is required when ingesting an existing file_id" in response.text, response.text
    assert rag_rig.calls() == ()
    assert _store_rows("") == before


def test_openai_file_id_attaches_to_an_existing_store_with_the_team_key(rag_rig: RagRig) -> None:
    store_id: Final = f"vs_existing_{uuid.uuid4().hex}"

    response: Final = _file_id_ingest(
        rag_rig.gateway, rag_rig.tenants.team_a_key, {**OPENAI, "vector_store_id": store_id}
    )

    assert _completed_store(rag_rig, response) == store_id
    assert rag_rig.signatures() == (("team", "POST", f"/v1/vector_stores/{store_id}/files", f"Bearer {TEAM_API_KEY}"),)


@pytest.mark.parametrize(
    ("knowledge_base", "error", "routes"),
    (
        (
            EMPTY_KNOWLEDGE_BASE,
            f"No data sources found for Knowledge Base {EMPTY_KNOWLEDGE_BASE}",
            (("POST", f"/knowledgebases/{EMPTY_KNOWLEDGE_BASE}/datasources/"),),
        ),
        (
            BUCKETLESS_KNOWLEDGE_BASE,
            f"Could not auto-detect S3 bucket for data source {WEB_DATA_SOURCE}",
            (
                ("POST", f"/knowledgebases/{BUCKETLESS_KNOWLEDGE_BASE}/datasources/"),
                ("GET", f"/knowledgebases/{BUCKETLESS_KNOWLEDGE_BASE}/datasources/{WEB_DATA_SOURCE}"),
            ),
        ),
    ),
    ids=("no_data_source", "no_s3_bucket"),
)
def test_bedrock_knowledge_base_the_caller_named_without_a_usable_data_source_fails_ingest_with_400(
    rag_rig: RagRig, knowledge_base: str, error: str, routes: tuple[tuple[str, str], ...]
) -> None:
    before: Final = _store_rows("")

    response: Final = _ingest(
        rag_rig.gateway,
        rag_rig.tenants.team_a_key,
        {"custom_llm_provider": "bedrock", "vector_store_id": knowledge_base},
    )

    assert response.status_code == 400, response.text
    assert error in response.text, response.text
    assert tuple((call.prefix, call.method, call.route) for call in rag_rig.calls()) == tuple(
        ("bedrock", method, route) for method, route in routes
    )
    assert _store_rows("") == before


def test_s3_vectors_bucket_name_the_caller_sent_is_rejected_with_400_after_embedding(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants
    before: Final = _store_rows("")
    request: Final = {
        "request": json.dumps(
            {
                "ingest_options": {
                    "name": "litellm-rag-ingest",
                    "embedding": {"model": tenants.shared_model},
                    "vector_store": {"custom_llm_provider": "s3_vectors", "vector_bucket_name": "ab", "dimension": 3},
                }
            }
        )
    }

    response: Final = rag_rig.gateway.request_multipart(
        "/v1/rag/ingest", request, {"file": DOCUMENT}, key=tenants.team_a_key
    )

    assert response.status_code == 400, response.text
    assert "Invalid vector_bucket_name 'ab'" in response.text, response.text
    assert rag_rig.signatures() == (("shared", "POST", "/v1/embeddings", f"Bearer {SHARED_API_KEY}"),)
    assert _store_rows("") == before


@dataclass(frozen=True, slots=True)
class KeylessRig:
    gateway: Gateway
    upstream: Wire
    key: str


@pytest.fixture(scope="module")
def keyless_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[KeylessRig]:
    directory: Final = tmp_path_factory.mktemp("rag_keyless_providers")
    with (
        gateway_from_environment() as admin_gateway,
        admin_gateway.scenario() as scenario,
        wire_server(_responder(threading.Event())) as upstream,
    ):
        key: Final = scenario.key()
        with owned_proxy(
            admin_gateway,
            directory,
            {"OPENAI_API_KEY": ENVIRONMENT_API_KEY, "OPENAI_API_BASE": f"{upstream.url}/environment/v1"},
            remove_environment=("GEMINI_API_KEY", "GOOGLE_API_KEY", "VERTEXAI_PROJECT"),
            workers=2,
        ) as gateway:
            yield KeylessRig(gateway, upstream, key)


@pytest.mark.parametrize(
    ("provider", "status", "error"),
    (
        ("gemini", 500, "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini File Search"),
        ("vertex_ai", 400, "vertex_project is required for Vertex AI RAG ingestion"),
    ),
    ids=("gemini_without_a_proxy_key", "vertex_ai_without_a_project"),
)
def test_ingest_status_says_whether_the_caller_or_the_proxy_must_fix_the_missing_setting(
    keyless_rig: KeylessRig, provider: str, status: int, error: str
) -> None:
    before: Final = _store_rows("")

    response: Final = _ingest(keyless_rig.gateway, keyless_rig.key, {"custom_llm_provider": provider})

    assert response.status_code == status, response.text
    assert error in response.text, response.text
    assert tuple(keyless_rig.upstream.drain()) == ()
    assert _store_rows("") == before


async def _jwt_chat_burst(rig: RagRig, token: str, model: str, count: int) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(base_url=str(rig.gateway.client.base_url), timeout=60, trust_env=False) as client:
        return tuple(
            await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json=_llm_body("/v1/chat/completions", model),
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    for _ in range(count)
                )
            )
        )


async def test_jwt_in_ten_teams_that_hold_the_model_always_resolves_the_lowest_team_id(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants
    token: Final = rag_rig.jwt(tenants.jwt_multi_member, *reversed(tenants.jwt_teams))

    responses: Final = await _jwt_chat_burst(rag_rig, token, tenants.jwt_team_model, 16)

    assert [response.status_code for response in responses] == [200] * 16, [response.text for response in responses]
    assert (
        rag_rig.signatures()
        == (("team", "POST", "/v1/chat/completions", f"Bearer {_jwt_team_key(min(tenants.jwt_teams))}"),) * 16
    )


def _prompt() -> str:
    return f"hello {uuid.uuid4().hex}"


def _sync_sdk_text(route: str, base_url: str, token: str, model: str, stream: bool) -> str:
    if route == "/v1/messages":
        messages_client: Final = anthropic.Anthropic(base_url=base_url, api_key=token, max_retries=0)
        request: Final = {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": _prompt()}]}
        if stream:
            return "".join(
                event.delta.text
                for event in messages_client.messages.create(**request, stream=True)
                if event.type == "content_block_delta" and event.delta.type == "text_delta"
            )
        message: Final = messages_client.messages.create(**request)
        return "".join(block.text for block in message.content if block.type == "text")
    client: Final = openai.OpenAI(base_url=f"{base_url}/v1", api_key=token, max_retries=0)
    if route == "/v1/responses":
        if stream:
            return "".join(
                event.delta
                for event in client.responses.create(model=model, input=_prompt(), stream=True)
                if event.type == "response.output_text.delta"
            )
        return client.responses.create(model=model, input=_prompt()).output_text
    chat_messages: Final = [{"role": "user", "content": _prompt()}]
    if stream:
        return "".join(
            chunk.choices[0].delta.content or ""
            for chunk in client.chat.completions.create(model=model, messages=chat_messages, stream=True)
            if chunk.choices
        )
    return client.chat.completions.create(model=model, messages=chat_messages).choices[0].message.content or ""


async def _async_sdk_text(route: str, base_url: str, token: str, model: str, stream: bool) -> str:
    if route == "/v1/messages":
        messages_client: Final = anthropic.AsyncAnthropic(base_url=base_url, api_key=token, max_retries=0)
        request: Final = {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": _prompt()}]}
        if stream:
            return "".join(
                [
                    event.delta.text
                    async for event in await messages_client.messages.create(**request, stream=True)
                    if event.type == "content_block_delta" and event.delta.type == "text_delta"
                ]
            )
        message: Final = await messages_client.messages.create(**request)
        return "".join(block.text for block in message.content if block.type == "text")
    client: Final = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=token, max_retries=0)
    if route == "/v1/responses":
        if stream:
            return "".join(
                [
                    event.delta
                    async for event in await client.responses.create(model=model, input=_prompt(), stream=True)
                    if event.type == "response.output_text.delta"
                ]
            )
        return (await client.responses.create(model=model, input=_prompt())).output_text
    chat_messages: Final = [{"role": "user", "content": _prompt()}]
    if stream:
        return "".join(
            [
                chunk.choices[0].delta.content or ""
                async for chunk in await client.chat.completions.create(
                    model=model, messages=chat_messages, stream=True
                )
                if chunk.choices
            ]
        )
    completion: Final = await client.chat.completions.create(model=model, messages=chat_messages)
    return completion.choices[0].message.content or ""


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.parametrize("client", ("sync", "async"))
@pytest.mark.parametrize("route", ("/v1/chat/completions", "/v1/messages", "/v1/responses"))
async def test_jwt_team_member_reaches_its_llm_routes_through_the_sdks(
    rag_rig: RagRig, route: str, client: Literal["sync", "async"], stream: bool
) -> None:
    tenants: Final = rag_rig.tenants
    base_url: Final = str(rag_rig.gateway.client.base_url).rstrip("/")
    token: Final = rag_rig.jwt(tenants.jwt_member, tenants.team_a)

    text: Final = (
        await asyncio.to_thread(_sync_sdk_text, route, base_url, token, tenants.team_a_model, stream)
        if client == "sync"
        else await _async_sdk_text(route, base_url, token, tenants.team_a_model, stream)
    )

    assert text == "integration response", text
    calls: Final = rag_rig.calls()
    assert [(call.prefix, call.authorization) for call in calls] == [("team", f"Bearer {TEAM_API_KEY}")], calls


def _batch_for(rig: RagRig, headers: Mapping[str, str]) -> tuple[str, str]:
    uploaded: Final = rig.gateway.client.post(
        "/v1/files", data={"purpose": "batch"}, files={"file": BATCH_INPUT}, headers=dict(headers)
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id: Final = string_value(object_value(uploaded.json())["id"])
    created: Final = rig.gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        headers=headers,
    )
    assert created.status_code == 200, created.text
    rig.upstream.drain()
    return file_id, string_value(object_value(created.json())["id"])


FOLLOW_UP_CALLS: Final = (
    ("GET", "/v1/files/{file_id}", "/v1/files/{file_id}"),
    ("GET", "/v1/files/{file_id}/content", "/v1/files/{file_id}/content"),
    ("DELETE", "/v1/files/{file_id}", "/v1/files/{file_id}"),
    ("GET", "/v1/files?purpose=batch&custom_llm_provider=openai", "/v1/files"),
    ("GET", "/v1/batches/{batch_id}", "/v1/batches/{batch_id}"),
    ("POST", "/v1/batches/{batch_id}/cancel", "/v1/batches/{batch_id}/cancel"),
)
FOLLOW_UP_CALL_IDS: Final = (
    "file_retrieve",
    "file_content",
    "file_delete",
    "file_list",
    "batch_retrieve",
    "batch_cancel",
)


def _bearer(key: str) -> Mapping[str, str]:
    return MappingProxyType({"Authorization": f"Bearer {key}"})


@pytest.mark.parametrize(("method", "path", "route"), FOLLOW_UP_CALLS, ids=FOLLOW_UP_CALL_IDS)
def test_files_and_batches_calls_by_a_key_restricted_under_the_team_wildcard_use_the_team_deployment(
    rag_rig: RagRig, method: str, path: str, route: str
) -> None:
    key: Final = rag_rig.tenants.wildcard_bare_key
    file_id, batch_id = _batch_for(rag_rig, _bearer(key))

    response: Final = rag_rig.gateway.request(method, path.format(file_id=file_id, batch_id=batch_id), key=key)

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        ("wildcard", method, route.format(file_id=file_id, batch_id=batch_id), f"Bearer {WILDCARD_API_KEY}"),
    )


def test_batch_list_by_a_key_restricted_under_the_team_wildcard_is_answered_by_the_proxy_without_an_upstream_call(
    rag_rig: RagRig,
) -> None:
    key: Final = rag_rig.tenants.wildcard_bare_key
    _batch_for(rag_rig, _bearer(key))

    response: Final = rag_rig.gateway.request("GET", "/v1/batches?custom_llm_provider=openai", key=key)

    assert response.status_code == 200, response.text
    assert object_value(response.json())["object"] == "list", response.text
    assert rag_rig.signatures() == ()


def _upload_batch_input(rig: RagRig, key: str) -> str:
    uploaded: Final = rig.gateway.request_multipart("/v1/files", {"purpose": "batch"}, {"file": BATCH_INPUT}, key=key)
    assert uploaded.status_code == 200, uploaded.text
    return string_value(object_value(uploaded.json())["id"])


def _create_batch(rig: RagRig, key: str, file_id: str) -> httpx.Response:
    return rig.gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        key=key,
    )


@pytest.mark.parametrize(
    ("team", "prefix", "api_key"),
    (("team_a", "team", TEAM_API_KEY), ("team_s", "shared", SHARED_API_KEY)),
    ids=("team_deployment", "accessible_deployment"),
)
def test_batch_over_the_key_tpm_limit_is_rejected_after_reading_its_input_where_it_was_uploaded(
    rag_rig: RagRig, team: Literal["team_a", "team_s"], prefix: str, api_key: str
) -> None:
    key: Final = rag_rig.scenario.key(team_id=getattr(rag_rig.tenants, team), tpm_limit=1)
    file_id: Final = _upload_batch_input(rag_rig, key)

    response: Final = _create_batch(rag_rig, key, file_id)

    assert response.status_code == 429, response.text
    bearer: Final = f"Bearer {api_key}"
    assert rag_rig.signatures() == (
        (prefix, "POST", "/v1/files", bearer),
        (prefix, "GET", f"/v1/files/{file_id}/content", bearer),
    )


def test_batch_whose_input_names_a_model_outside_the_key_allowlist_is_rejected_before_creation(
    rag_rig: RagRig,
) -> None:
    key: Final = rag_rig.tenants.restricted_key
    file_id: Final = _upload_batch_input(rag_rig, key)

    response: Final = _create_batch(rag_rig, key, file_id)

    assert response.status_code == 403, response.text
    assert BARE_WILDCARD_MODEL in response.text, response.text
    bearer: Final = f"Bearer {SHARED_API_KEY}"
    assert rag_rig.signatures() == (
        ("shared", "POST", "/v1/files", bearer),
        ("shared", "GET", f"/v1/files/{file_id}/content", bearer),
    )


def test_azure_files_upload_by_a_key_restricted_under_the_team_azure_wildcard_uses_the_team_deployment(
    rag_rig: RagRig,
) -> None:
    response: Final = rag_rig.gateway.request_multipart(
        "/v1/files",
        {"purpose": "assistants", "custom_llm_provider": "azure"},
        {"file": DOCUMENT},
        key=rag_rig.tenants.azure_restricted_key,
    )

    assert response.status_code == 200, response.text
    assert tuple((call.prefix, call.method, call.route, call.azure_key) for call in rag_rig.calls()) == (
        ("azure", "POST", "/openai/files", WILDCARD_API_KEY),
    )


def test_query_without_a_vector_store_id_is_rejected_before_search(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = _query(rag_rig.gateway, tenants.team_a_key, tenants.team_a_model, OPENAI)

    assert response.status_code == 400, response.text
    assert "retrieval_config must contain 'vector_store_id'" in response.text, response.text
    assert rag_rig.calls() == ()


def _forwarding(key: str) -> Mapping[str, str]:
    return MappingProxyType({"Authorization": f"Bearer {key}", "x-api-key": FORWARDED_API_KEY})


@pytest.mark.parametrize(
    ("stored", "stream", "search"),
    (
        (None, False, ("team", FORWARDED_API_KEY)),
        (None, True, ("team", FORWARDED_API_KEY)),
        ("own_key", False, ("caller", CALLER_API_KEY)),
        ("null_key", False, ("team", FORWARDED_API_KEY)),
    ),
    ids=("unmanaged_json", "unmanaged_stream", "managed_own_key", "managed_null_key"),
)
def test_forwarded_provider_key_on_query_wins_over_team_provider_credentials(
    rag_rig: RagRig, stored: Literal["own_key", "null_key"] | None, stream: bool, search: tuple[str, str]
) -> None:
    tenants: Final = rag_rig.tenants
    with rag_rig.gateway.scenario() as scenario:
        store_id: Final = (
            f"vs_unmanaged_{uuid.uuid4().hex}"
            if stored is None
            else _managed_store(
                rag_rig,
                scenario,
                {"api_key": CALLER_API_KEY, "api_base": f"{rag_rig.upstream.url}/caller/v1"}
                if stored == "own_key"
                else {"api_key": None},
            )
        )

        response: Final = rag_rig.gateway.request(
            "POST",
            "/v1/rag/query",
            {
                "model": tenants.team_a_model,
                "messages": [{"role": "user", "content": f"query forwarded key {uuid.uuid4().hex}"}],
                "retrieval_config": {**OPENAI, "top_k": 1, "vector_store_id": store_id},
                **({"stream": True} if stream else {}),
            },
            headers=_forwarding(tenants.team_a_key),
        )

        assert response.status_code == 200, response.text
        assert "integration response" in response.text, response.text
        assert rag_rig.signatures() == (
            _search_signature(search[0], search[1], store_id),
            _chat_signature("team", FORWARDED_API_KEY),
        )


def test_forwarded_provider_key_on_ingest_wins_over_team_provider_credentials(rag_rig: RagRig) -> None:
    tenants: Final = rag_rig.tenants

    response: Final = rag_rig.gateway.client.post(
        "/v1/rag/ingest",
        data=dict(_ingest_request("litellm-rag-forwarded-key", OPENAI)),
        files={"file": DOCUMENT},
        headers=dict(_forwarding(tenants.team_a_key)),
    )

    store_id: Final = _completed_store(rag_rig, response)
    assert rag_rig.signatures() == _ingest_signatures("team", FORWARDED_API_KEY, store_id)


@pytest.mark.parametrize(
    ("caller", "prefix"),
    (("team_a_key", "team"), ("wildcard_bare_key", "wildcard")),
    ids=("team_key", "restricted_wildcard_key"),
)
def test_forwarded_provider_key_on_files_and_batches_wins_over_team_provider_credentials(
    rag_rig: RagRig, caller: Literal["team_a_key", "wildcard_bare_key"], prefix: str
) -> None:
    headers: Final = _forwarding(getattr(rag_rig.tenants, caller))
    uploaded: Final = rag_rig.gateway.client.post(
        "/v1/files", data={"purpose": "batch"}, files={"file": BATCH_INPUT}, headers=dict(headers)
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id: Final = string_value(object_value(uploaded.json())["id"])

    created: Final = rag_rig.gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        headers=headers,
    )

    assert created.status_code == 200, created.text
    calls: Final = rag_rig.calls()
    assert [(call.prefix, call.authorization) for call in calls] == [
        (prefix, f"Bearer {FORWARDED_API_KEY}")
    ] * len(calls), calls
    assert {(call.method, call.route) for call in calls} >= {("POST", "/v1/files"), ("POST", "/v1/batches")}, calls


@pytest.mark.parametrize("stray_claim", (7, None), ids=("int", "null"))
@pytest.mark.parametrize("route", ("/v1/chat/completions", "/v1/rag/query"))
def test_jwt_team_claim_with_a_non_string_entry_still_resolves_the_string_team(
    rag_rig: RagRig, route: str, stray_claim: JsonValue
) -> None:
    tenants: Final = rag_rig.tenants
    token: Final = rag_rig.jwt(tenants.jwt_member, tenants.team_a, stray_claim)
    store_id: Final = f"vs_unmanaged_{uuid.uuid4().hex}"

    response: Final = (
        _query(rag_rig.gateway, token, tenants.team_a_model, {**OPENAI, "vector_store_id": store_id})
        if route == "/v1/rag/query"
        else rag_rig.gateway.request("POST", route, _llm_body(route, tenants.team_a_model), key=token)
    )

    assert response.status_code == 200, response.text
    search: Final = (_search_signature("team", TEAM_API_KEY, store_id),) if route == "/v1/rag/query" else ()
    assert rag_rig.signatures() == (*search, _chat_signature("team", TEAM_API_KEY))


@pytest.mark.parametrize("stray_claim", (7, None), ids=("int", "null"))
def test_jwt_header_team_outside_a_claim_with_a_non_string_entry_is_denied_with_403(
    rag_rig: RagRig, stray_claim: JsonValue
) -> None:
    tenants: Final = rag_rig.tenants
    header_team: Final = f"integration-rag-unknown-team-{uuid.uuid4().hex}"

    response: Final = rag_rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        _llm_body("/v1/chat/completions", tenants.team_a_model),
        key=rag_rig.jwt(tenants.jwt_member, tenants.team_a, stray_claim),
        headers={"x-litellm-team-id": header_team},
    )

    assert response.status_code == 403, response.text
    assert f"x-litellm-team-id '{header_team}' does not resolve to a team id or a unique team alias" in response.text, (
        response.text
    )
    assert f"'{tenants.team_a}'" in response.text, response.text
    assert rag_rig.calls() == ()


@pytest.mark.parametrize(("method", "path", "route"), FOLLOW_UP_CALLS, ids=FOLLOW_UP_CALL_IDS)
@pytest.mark.parametrize(
    ("caller", "prefix"),
    (("team_a_key", "team"), ("wildcard_bare_key", "wildcard")),
    ids=("team_key", "restricted_wildcard_key"),
)
def test_forwarded_provider_key_on_files_and_batches_follow_up_calls_replaces_the_team_deployment_key(
    rag_rig: RagRig,
    caller: Literal["team_a_key", "wildcard_bare_key"],
    prefix: str,
    method: str,
    path: str,
    route: str,
) -> None:
    headers: Final = _forwarding(getattr(rag_rig.tenants, caller))
    file_id, batch_id = _batch_for(rag_rig, headers)

    response: Final = rag_rig.gateway.request(method, path.format(file_id=file_id, batch_id=batch_id), headers=headers)

    assert response.status_code == 200, response.text
    assert rag_rig.signatures() == (
        (prefix, method, route.format(file_id=file_id, batch_id=batch_id), f"Bearer {FORWARDED_API_KEY}"),
    )


@pytest.mark.parametrize(
    ("x_api_key", "prefix", "api_key"),
    (
        (("",), "wildcard", WILDCARD_API_KEY),
        (("f" * 5120,), "wildcard", "f" * 5120),
        ((FORWARDED_API_KEY, FORWARDED_API_KEY), "wildcard", FORWARDED_API_KEY),
    ),
    ids=("empty", "five_kb", "repeated"),
)
def test_blank_oversized_or_repeated_forwarded_provider_key_on_files_and_batches(
    rag_rig: RagRig, x_api_key: tuple[str, ...], prefix: str, api_key: str
) -> None:
    headers: Final = (
        ("Authorization", f"Bearer {rag_rig.tenants.wildcard_bare_key}"),
        *(("x-api-key", value) for value in x_api_key),
    )
    uploaded: Final = rag_rig.gateway.client.post(
        "/v1/files", data={"purpose": "batch"}, files={"file": BATCH_INPUT}, headers=headers
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id: Final = string_value(JSON_OBJECT.validate_json(uploaded.content)["id"])

    created: Final = rag_rig.gateway.client.post(
        "/v1/batches",
        json={"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        headers=headers,
    )

    assert created.status_code == 200, created.text
    calls: Final = rag_rig.calls()
    assert [(call.prefix, call.authorization) for call in calls] == [(prefix, f"Bearer {api_key}")] * len(calls), calls
    assert {(call.method, call.route) for call in calls} == {
        ("POST", "/v1/files"),
        ("GET", f"/v1/files/{file_id}/content"),
        ("POST", "/v1/batches"),
    }, calls


def _forwarded_ingest(rig: RagRig, vector_store: Mapping[str, JsonValue]) -> httpx.Response:
    return rig.gateway.client.post(
        "/v1/rag/ingest",
        data=dict(_ingest_request("litellm-rag-forwarded-key", vector_store)),
        files={"file": DOCUMENT},
        headers=dict(_forwarding(rig.tenants.team_a_key)),
    )


def test_forwarded_provider_key_on_ingest_never_overrides_a_named_credential(rag_rig: RagRig) -> None:
    store_id: Final = _completed_store(
        rag_rig, _forwarded_ingest(rag_rig, {**OPENAI, "litellm_credential_name": rag_rig.tenants.credential_name})
    )

    assert rag_rig.signatures() == _ingest_signatures("credential", CREDENTIAL_API_KEY, store_id)


@pytest.mark.parametrize(
    ("stored", "prefix", "api_key"),
    (("own_key", "caller", CALLER_API_KEY), ("null_key", "team", FORWARDED_API_KEY)),
    ids=("own_key", "null_key"),
)
def test_forwarded_provider_key_on_ingest_into_a_managed_store_yields_only_to_the_store_key(
    rag_rig: RagRig, stored: Literal["own_key", "null_key"], prefix: str, api_key: str
) -> None:
    with rag_rig.gateway.scenario() as scenario:
        store_id: Final = _managed_store(
            rag_rig,
            scenario,
            {"api_key": CALLER_API_KEY, "api_base": f"{rag_rig.upstream.url}/caller/v1"}
            if stored == "own_key"
            else {"api_key": None},
        )

        response: Final = _forwarded_ingest(rag_rig, {**OPENAI, "vector_store_id": store_id})

        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["vector_store_id"] == store_id, response.text
        bearer: Final = f"Bearer {api_key}"
        assert rag_rig.signatures() == (
            (prefix, "POST", "/v1/files", bearer),
            (prefix, "POST", f"/v1/vector_stores/{store_id}/files", bearer),
        )


def test_forwarded_provider_key_never_reaches_a_gemini_ingest(rag_rig: RagRig) -> None:
    store_id: Final = _completed_store(rag_rig, _forwarded_ingest(rag_rig, {"custom_llm_provider": "gemini"}))

    requests: Final = rag_rig.upstream.drain()
    store_name: Final = store_id.removeprefix("fileSearchStores/")
    assert tuple(
        (call.prefix, call.method, call.route, call.google_key, call.authorization)
        for call in (_call(request) for request in requests)
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
    assert [request.headers.get("x-api-key") for request in requests] == [None] * len(requests), requests
