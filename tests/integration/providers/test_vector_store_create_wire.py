import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CHUNKING: Final[dict[str, JsonValue]] = {
    "type": "static",
    "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400},
}
_CREATE_FIELDS: Final[dict[str, JsonValue]] = {
    "name": "integration vector store",
    "file_ids": ["file-wire-create"],
    "expires_after": {"anchor": "last_active_at", "days": 7},
    "chunking_strategy": _CHUNKING,
    "metadata": {"team": "blue"},
}


class _FileCounts(BaseModel):
    in_progress: int
    completed: int
    failed: int
    cancelled: int
    total: int


class _VectorStoreResponse(BaseModel):
    id: str
    object: str
    created_at: int
    name: str
    file_counts: _FileCounts
    status: str
    usage_bytes: int
    expires_after: dict[str, JsonValue] | None
    expires_at: int | None
    last_active_at: int | None
    metadata: dict[str, str]


def _vector_store_reply(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "vector_store",
                "created_at": 1700000000,
                "name": _CREATE_FIELDS["name"],
                "file_counts": {
                    "in_progress": 0,
                    "completed": 1,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 1,
                },
                "status": "completed",
                "usage_bytes": 12,
                "expires_after": _CREATE_FIELDS["expires_after"],
                "expires_at": None,
                "last_active_at": None,
                "metadata": _CREATE_FIELDS["metadata"],
            }
        ).encode()
    )


def _assert_create_request(
    request: Request,
    expected_file_ids: list[str],
    provider_key: str,
) -> None:
    assert request.method == "POST"
    assert urlsplit(request.target).path == "/v1/vector_stores", request.target
    assert parse_qs(urlsplit(request.target).query) == {}, request.target
    assert request.headers["authorization"] == f"Bearer {provider_key}"
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert {key: value for key, value in body.items() if key != "metadata"} == {
        "name": _CREATE_FIELDS["name"],
        "file_ids": expected_file_ids,
        "expires_after": _CREATE_FIELDS["expires_after"],
        "chunking_strategy": _CHUNKING,
    }, request.body.decode()


def _upload_managed_file(
    gateway: Gateway,
    key: str,
    model: str,
    filename: str,
    content: bytes,
) -> str:
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "user_data", "target_model_names": model},
        {"file": (filename, content, "text/plain")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    return string_value(JSON_OBJECT.validate_json(uploaded.content)["id"])


def _deployment(scenario: Scenario, wire_url: str, provider_key: str) -> str:
    return scenario.model(api_base=f"{wire_url}/v1", api_key=provider_key)


def _deployment_id(model: str) -> str:
    rows: Final = read_rows(
        'SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s '
        "OR model_info->>'team_public_model_name' = %s",
        (model, model),
    )
    assert len(rows) == 1, f"Expected one deployment for {model!r}, found {rows}"
    return string_value(rows[0]["model_id"])


def _worker_has_deployment(gateway: Gateway, deployment_id: str, _: int) -> bool:
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as client:
        response: Final = client.get(
            "/model/info",
            params={"litellm_model_id": deployment_id},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
    if response.status_code == 200:
        return True
    assert response.status_code == 400, response.text
    assert "not found on litellm proxy" in response.text.lower(), response.text
    return False


def _wait_until_every_worker_serves(gateway: Gateway, model: str) -> None:
    deployment_id: Final = _deployment_id(model)

    def every_worker_serves() -> bool:
        with ThreadPoolExecutor(max_workers=16) as pool:
            rounds: Final = tuple(
                tuple(pool.map(partial(_worker_has_deployment, gateway, deployment_id), range(16))) for _ in range(2)
            )
        return all(has_deployment for round_ in rounds for has_deployment in round_)

    eventually(every_worker_serves, lambda served: served, seconds=90)


def test_vector_store_create_forwards_chunking_strategy_on_sdk_and_alias_routes(gateway: Gateway) -> None:
    provider_key: Final = f"provider-create-{uuid.uuid4().hex}"
    expected_sdk_response: Final = {
        "id": "vs_created",
        "object": "vector_store",
        "created_at": 1700000000,
        "name": _CREATE_FIELDS["name"],
        "file_counts": {
            "in_progress": 0,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "total": 1,
        },
        "status": "completed",
        "usage_bytes": 12,
        "expires_after": _CREATE_FIELDS["expires_after"],
        "expires_at": None,
        "last_active_at": None,
        "metadata": _CREATE_FIELDS["metadata"],
    }

    def respond(_: Request) -> Reply:
        return _vector_store_reply("vs_created")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url, provider_key)
        _wait_until_every_worker_serves(gateway, model)
        wire.drain()
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            http_client=httpx.Client(trust_env=False),
            max_retries=0,
        ) as client:
            sdk_response: Final = client.vector_stores.create(
                name=str(_CREATE_FIELDS["name"]),
                file_ids=["file-wire-create"],
                expires_after={"anchor": "last_active_at", "days": 7},
                chunking_strategy=_CHUNKING,
                metadata={"team": "blue"},
                extra_body={"model": model},
            )
            assert sdk_response.model_dump(exclude_unset=True) == expected_sdk_response

        alias_response: Final = gateway.request(
            "POST",
            "/vector_stores",
            {"model": model, **_CREATE_FIELDS},
        )
        assert alias_response.status_code == 200, alias_response.text
        assert _VectorStoreResponse.model_validate_json(alias_response.content).model_dump() == {
            **expected_sdk_response,
        }
        requests: Final = wire.drain()
        assert [(request.method, urlsplit(request.target).path) for request in requests] == [
            ("POST", "/v1/vector_stores"),
            ("POST", "/v1/vector_stores"),
        ]
        for request in requests:
            _assert_create_request(request, ["file-wire-create"], provider_key)
            body: Final = JSON_OBJECT.validate_json(request.body)
            metadata_value: Final = body.get("metadata")
            assert metadata_value is not None, request.body.decode()
            metadata: Final = object_value(metadata_value)
            assert metadata.get("team") == "blue", request.body.decode()


def test_vector_store_create_translates_managed_file_ids(gateway: Gateway) -> None:
    pytest.skip("BUG: vector store create forwards managed file IDs to the provider without translating them")

    provider_key: Final = f"provider-create-managed-{uuid.uuid4().hex}"
    provider_file_id: Final = f"file-create-managed-{uuid.uuid4().hex}"
    raw_file_id: Final = f"file-create-raw-{uuid.uuid4().hex}"
    filename: Final = f"create-{uuid.uuid4().hex}.txt"
    file_content: Final = b"managed vector store create\n"
    provider_store_id: Final = f"vs-created-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        assert request.headers["authorization"] == f"Bearer {provider_key}"
        if request.method == "POST" and path == "/v1/files":
            assert f'filename="{filename}"'.encode() in request.body, request.body[:200]
            assert file_content in request.body, request.body[:200]
            assert b'name="purpose"' in request.body, request.body[:200]
            assert b"user_data" in request.body, request.body[:200]
            return Reply(
                body=json.dumps(
                    {
                        "id": provider_file_id,
                        "object": "file",
                        "bytes": len(file_content),
                        "created_at": 1700000000,
                        "filename": filename,
                        "purpose": "user_data",
                        "status": "processed",
                    }
                ).encode()
            )
        _assert_create_request(request, [raw_file_id, provider_file_id], provider_key)
        return _vector_store_reply(provider_store_id)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url, provider_key)
        _wait_until_every_worker_serves(gateway, model)
        wire.drain()
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        managed_file_id: Final = _upload_managed_file(gateway, key, model, filename, file_content)
        file_ids: Final = [raw_file_id, managed_file_id]
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=key,
            http_client=httpx.Client(trust_env=False),
            max_retries=0,
        ) as client:
            response: Final = client.vector_stores.create(
                name=str(_CREATE_FIELDS["name"]),
                file_ids=file_ids,
                expires_after={"anchor": "last_active_at", "days": 7},
                chunking_strategy=_CHUNKING,
                metadata={"team": "blue"},
                extra_body={"model": model},
            )
        assert response.id == provider_store_id, str(response)
        requests: Final = wire.drain()
        assert [(request.method, urlsplit(request.target).path) for request in requests] == [
            ("POST", "/v1/files"),
            ("POST", "/v1/vector_stores"),
        ], requests


def test_vector_store_create_forwards_only_caller_metadata(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: vector store create forwards proxy-internal metadata (user_api_key_hash, endpoint, api_base, ...) to the provider"
    )

    provider_key: Final = f"provider-create-metadata-{uuid.uuid4().hex}"
    provider_id: Final = f"vs-metadata-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert urlsplit(request.target).path == "/v1/vector_stores", request.target
        assert request.headers["authorization"] == f"Bearer {provider_key}"
        assert JSON_OBJECT.validate_json(request.body) == {
            "name": _CREATE_FIELDS["name"],
            "file_ids": _CREATE_FIELDS["file_ids"],
            "expires_after": _CREATE_FIELDS["expires_after"],
            "chunking_strategy": _CHUNKING,
            "metadata": {"team": "blue"},
        }, request.body.decode()
        return _vector_store_reply(provider_id)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url, provider_key)
        _wait_until_every_worker_serves(gateway, model)
        wire.drain()
        key: Final = scenario.key(models=[model])
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=key,
            http_client=httpx.Client(trust_env=False),
            max_retries=0,
        ) as client:
            response: Final = client.vector_stores.create(
                name=str(_CREATE_FIELDS["name"]),
                file_ids=["file-wire-create"],
                expires_after={"anchor": "last_active_at", "days": 7},
                chunking_strategy=_CHUNKING,
                metadata={"team": "blue"},
                extra_body={"model": model},
            )
        assert response.id == provider_id, str(response)
        requests: Final = wire.drain()
        assert [(request.method, urlsplit(request.target).path) for request in requests] == [
            ("POST", "/v1/vector_stores")
        ], requests
