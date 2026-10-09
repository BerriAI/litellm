from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import jwt
import httpx
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, gateway_from_environment, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

TEAM_API_KEY: Final = "integration-rag-team-provider-key"
JWT_KEY_ID: Final = "integration-rag-team-provider-jwt"
VECTOR_STORE_ID: Final = f"vs_rag_team_provider_{uuid.uuid4().hex}"


class RagRig:
    def __init__(self, gateway: Gateway, upstream: Wire, signing_key: rsa.RSAPrivateKey) -> None:
        self.gateway: Final = gateway
        self.upstream: Final = upstream
        self.signing_key: Final = signing_key

    def jwt(self, subject: str, team_id: str) -> str:
        issued_at: Final = int(time.time())
        return jwt.encode(
            {"sub": subject, "groups": [team_id], "iat": issued_at, "exp": issued_at + 300},
            self.signing_key,
            algorithm="RS256",
            headers={"kid": JWT_KEY_ID},
        )


def _json_body(request: Request) -> Mapping[str, JsonValue]:
    if not request.body:
        return MappingProxyType({})
    return object_value(json.loads(request.body))


def _provider_reply(request: Request) -> Reply:
    if request.method == "POST" and request.target == "/v1/vector_stores":
        if _json_body(request).get("name") == "rag-upstream-401":
            return Reply(
                status=401,
                body=b'{"error":{"message":"unauthorized","type":"invalid_request_error","code":"invalid_api_key"}}',
            )
        return Reply(
            body=json.dumps(
                {
                    "id": VECTOR_STORE_ID,
                    "object": "vector_store",
                    "created_at": int(time.time()),
                    "name": "litellm-rag-ingest",
                    "usage_bytes": 0,
                    "status": "completed",
                }
            ).encode()
        )
    if request.method == "POST" and request.target == "/v1/files":
        return Reply(
            body=json.dumps(
                {
                    "id": "file_rag_team_provider",
                    "object": "file",
                    "bytes": 12,
                    "created_at": int(time.time()),
                    "filename": "document.txt",
                    "purpose": "assistants",
                }
            ).encode()
        )
    if request.method == "POST" and request.target == f"/v1/vector_stores/{VECTOR_STORE_ID}/files":
        return Reply(
            body=json.dumps(
                {
                    "id": "file_rag_team_provider",
                    "object": "vector_store.file",
                    "created_at": int(time.time()),
                    "vector_store_id": VECTOR_STORE_ID,
                    "status": "completed",
                    "last_error": None,
                }
            ).encode()
        )
    if request.method == "POST" and request.target == f"/v1/vector_stores/{VECTOR_STORE_ID}/search":
        body: Final = _json_body(request)
        query: Final = body.get("query", "")
        return Reply(
            body=json.dumps(
                {
                    "object": "vector_store.search_results.page",
                    "search_query": query,
                    "data": [],
                    "has_more": False,
                    "next_page": None,
                }
            ).encode()
        )
    if request.method == "POST" and request.target == "/v1/chat/completions":
        body: Final = _json_body(request)
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "integration-rag-team-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "integration response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )
    return Reply(status=404, body=b'{"error":{"message":"unexpected upstream route"}}')


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
    }
    path: Final = directory / "rag_team_provider_credentials.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


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

    directory: Final = tmp_path_factory.mktemp("rag_team_provider_credentials")
    config: Final = _proxy_config(directory)
    with (
        gateway_from_environment() as admin_gateway,
        wire_server(_provider_reply) as upstream,
        wire_server(jwks_reply) as jwks,
        owned_proxy(
            admin_gateway,
            directory,
            {"JWT_PUBLIC_KEY_URL": jwks.url},
            config=config,
            remove_environment=("OPENAI_API_KEY", "OPENAI_API_BASE"),
        ) as gateway,
    ):
        yield RagRig(gateway, upstream, signing_key)


def _team_model(scenario: Scenario, rig: RagRig) -> tuple[str, str]:
    model_name: Final = f"integration-rag-team-{uuid.uuid4().hex}"
    team_id: Final = scenario.team(models=[model_name])
    created: Final = rig.gateway.post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": TEAM_API_KEY,
                "api_base": f"{rig.upstream.url}/v1",
            },
            "model_info": {"team_id": team_id},
        },
    )
    model_info: Final = object_value(created["model_info"])
    scenario.cleanups.callback(scenario.delete_model, str(model_info["id"]))
    return model_name, team_id


def _ingest(
    gateway: Gateway,
    key: str,
    *,
    name: str = "litellm-rag-ingest",
) -> httpx.Response:
    return gateway.request_multipart(
        "/v1/rag/ingest",
        fields={
            "request": json.dumps({"ingest_options": {"name": name, "vector_store": {"custom_llm_provider": "openai"}}})
        },
        files={"file": ("document.txt", b"team provider credentials", "text/plain")},
        key=key,
    )


def _query(gateway: Gateway, key: str, model: str, team_store_id: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/rag/query",
        {
            "model": model,
            "messages": [{"role": "user", "content": "query team vector store"}],
            "retrieval_config": {
                "vector_store_id": team_store_id,
                "custom_llm_provider": "openai",
                "top_k": 1,
            },
        },
        key=key,
    )


def _assert_team_authorization(requests: tuple[Request, ...], paths: tuple[str, ...]) -> None:
    observed: Final = tuple(
        (request.target, request.headers.get("authorization")) for request in requests if request.target in paths
    )
    expected: Final = tuple((path, f"Bearer {TEAM_API_KEY}") for path in paths)
    assert observed == expected


def test_viewer_ingest_and_team_and_jwt_query_use_team_provider_credentials(rag_rig: RagRig) -> None:
    with rag_rig.gateway.scenario() as scenario:
        model, team_id = _team_model(scenario, rag_rig)
        viewer_id: Final = scenario.user(user_role="internal_user_viewer")
        rag_rig.gateway.post(
            "/team/member_add",
            {"team_id": team_id, "member": {"user_id": viewer_id, "role": "user"}},
        )
        viewer_key: Final = scenario.key(user_id=viewer_id, team_id=team_id, models=[model])
        team_key: Final = scenario.key(team_id=team_id, models=[model])
        jwt_user: Final = scenario.member(team_id)

        ingest_response: Final = _ingest(rag_rig.gateway, viewer_key)
        assert ingest_response.status_code == 200, ingest_response.text
        ingest_body: Final = object_value(ingest_response.json())
        assert ingest_body["status"] == "completed", ingest_response.text
        store_id: Final = string_value(ingest_body["vector_store_id"])
        _assert_team_authorization(
            rag_rig.upstream.drain(),
            (
                "/v1/vector_stores",
                "/v1/files",
                f"/v1/vector_stores/{store_id}/files",
            ),
        )

        team_query: Final = _query(rag_rig.gateway, team_key, model, store_id)
        team_query_requests: Final = rag_rig.upstream.drain()
        assert team_query.status_code == 200, (
            f"{team_query.text}; upstream requests: "
            f"{tuple((request.target, request.headers.get('authorization')) for request in team_query_requests)}"
        )
        _assert_team_authorization(
            team_query_requests,
            (f"/v1/vector_stores/{store_id}/search",),
        )

        jwt_query: Final = _query(
            rag_rig.gateway,
            rag_rig.jwt(jwt_user, team_id),
            model,
            store_id,
        )
        assert jwt_query.status_code == 200, jwt_query.text
        _assert_team_authorization(
            rag_rig.upstream.drain(),
            (f"/v1/vector_stores/{store_id}/search",),
        )


def test_upstream_ingest_401_is_returned_without_success_response(rag_rig: RagRig) -> None:
    with rag_rig.gateway.scenario() as scenario:
        model, team_id = _team_model(scenario, rag_rig)
        team_key: Final = scenario.key(team_id=team_id, models=[model])

        response: Final = _ingest(rag_rig.gateway, team_key, name="rag-upstream-401")

        assert response.status_code == 401, response.text
        request: Final = rag_rig.upstream.drain()
        assert tuple(
            (item.target, item.headers.get("authorization")) for item in request if item.target == "/v1/vector_stores"
        ) == (("/v1/vector_stores", f"Bearer {TEAM_API_KEY}"),)
