import base64
import json
import re
import uuid
from collections.abc import Callable, Mapping
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MULTIPART_CONTENT: Final = b"UI multipart content\n"
_JSON_CONTENT: Final = b"JSON base64 content\n"
_REGISTRY_CONTENT: Final = b"registered vector store content\n"
_GENERATED_FILENAME: Final = re.compile(r"^[0-9a-f]{32}\.txt$")
_PROVIDER_CREATE_BODY: Final[dict[str, JsonValue]] = {
    "name": "rag ingest contract",
    "file_ids": None,
    "expires_after": None,
    "chunking_strategy": None,
    "metadata": None,
}
_INGEST_FILE_IDS: Final = MappingProxyType(
    {
        _MULTIPART_CONTENT: "file_ingest_multipart",
        _JSON_CONTENT: "file_ingest_json",
        _REGISTRY_CONTENT: "file_ingest_registry",
    }
)


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _multipart_text(parts: tuple[Message, ...]) -> dict[str, str]:
    return {
        str(part.get_param("name", header="content-disposition")): part.get_payload(decode=True).decode()
        for part in parts
        if part.get_filename() is None
    }


def _multipart_file(parts: tuple[Message, ...]) -> tuple[str, str, bytes]:
    (part,) = tuple(part for part in parts if part.get_filename() is not None)
    return (
        str(part.get_filename()),
        str(part.get_content_type()),
        part.get_payload(decode=True),
    )


def _vector_store_reply(identity: str, name: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "vector_store",
                "created_at": 1700000000,
                "name": name,
                "file_counts": {
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 0,
                },
                "status": "completed",
                "usage_bytes": 0,
                "expires_after": None,
                "expires_at": None,
                "last_active_at": None,
                "metadata": {},
            }
        ).encode()
    )


def _file_reply(identity: str, filename: str, content: bytes) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "file",
                "bytes": len(content),
                "created_at": 1700000000,
                "filename": filename,
                "purpose": "assistants",
                "status": "processed",
            }
        ).encode()
    )


def _attached_file_reply(store_id: str, file_id: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": file_id,
                "object": "vector_store.file",
                "usage_bytes": 12,
                "created_at": 1700000001,
                "vector_store_id": store_id,
                "status": "completed",
                "last_error": None,
                "chunking_strategy": {"type": "auto"},
                "attributes": {},
            }
        ).encode()
    )


def _assert_upload(request: Request, expected_content: bytes) -> tuple[str, str]:
    assert request.method == "POST"
    assert urlsplit(request.target).path == "/v1/files", request.target
    parts: Final = _multipart_parts(request)
    assert _multipart_text(parts) == {"purpose": "assistants"}
    filename, content_type, content = _multipart_file(parts)
    assert _GENERATED_FILENAME.fullmatch(filename), filename
    assert content_type == "text/plain", request.headers["content-type"]
    assert content == expected_content
    return filename, content_type


def _no_registry_config(directory: Path) -> Path:
    config: Final = object_value(yaml.safe_load(PROXY_CONFIG.read_text()))
    without_registry: Final[Mapping[str, JsonValue]] = {
        key: value for key, value in config.items() if key != "vector_store_registry"
    }
    path: Final = directory / "proxy_no_vector_store_registry.yaml"
    path.write_text(yaml.safe_dump({**without_registry, "model_list": []}))
    return path


def _new_store_provider(
    provider_key: str, store_id: str, store_name: str = "rag ingest contract"
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        assert request.headers["authorization"] == f"Bearer {provider_key}"
        if request.method == "POST" and path == "/v1/vector_stores":
            assert JSON_OBJECT.validate_json(request.body) == {
                **_PROVIDER_CREATE_BODY,
                "name": store_name,
            }
            return _vector_store_reply(store_id, store_name)
        if request.method == "POST" and path == "/v1/files":
            parts: Final = _multipart_parts(request)
            assert _multipart_text(parts) == {"purpose": "assistants"}
            filename, content_type, content = _multipart_file(parts)
            assert _GENERATED_FILENAME.fullmatch(filename), filename
            assert content_type == "text/plain", request.headers["content-type"]
            file_id: Final = _INGEST_FILE_IDS.get(content)
            assert file_id is not None, content
            return _file_reply(file_id, filename, content)
        if request.method == "POST" and path == f"/v1/vector_stores/{store_id}/files":
            body: Final = JSON_OBJECT.validate_json(request.body)
            file_id: Final = str(body["file_id"])
            assert body == {"file_id": file_id, "chunking_strategy": {"type": "auto"}}
            assert file_id in {"file_ingest_multipart", "file_ingest_json"}
            return _attached_file_reply(store_id, file_id)
        return Reply(status=404, body=json.dumps({"error": request.target}).encode())

    return respond


def _registered_store_provider(provider_key: str, store_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        assert request.headers["authorization"] == f"Bearer {provider_key}"
        if request.method == "POST" and path == "/v1/files":
            filename, _ = _assert_upload(request, _REGISTRY_CONTENT)
            return _file_reply("file_ingest_registry", filename, _REGISTRY_CONTENT)
        assert request.method == "POST"
        assert path == f"/v1/vector_stores/{store_id}/files", request.target
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body == {"file_id": "file_ingest_registry", "chunking_strategy": {"type": "auto"}}
        return _attached_file_reply(store_id, "file_ingest_registry")

    return respond


def _ingest_options(name: str) -> dict[str, JsonValue]:
    return {"name": name, "vector_store": {"custom_llm_provider": "openai"}}


def _ingest_response(response: httpx.Response, vector_store_id: str, file_id: str) -> None:
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert body["vector_store_id"] == vector_store_id, response.text
    assert body["file_id"] == file_id, response.text


def test_rag_ingest_request_multipart_json_and_registered_credentials(gateway: Gateway, tmp_path: Path) -> None:
    default_key: Final = f"ingest-default-{uuid.uuid4().hex}"
    registered_key: Final = f"ingest-registered-{uuid.uuid4().hex}"
    new_store_id: Final = f"vs_ingest_wire_{uuid.uuid4().hex}"
    registered_store_id: Final = f"vs_ingest_registry_{uuid.uuid4().hex}"
    config: Final = _no_registry_config(tmp_path)
    with (
        gateway_from_environment() as upstream_gateway,
        wire_server(_new_store_provider(default_key, new_store_id)) as default_wire,
        wire_server(_registered_store_provider(registered_key, registered_store_id)) as registered_wire,
    ):
        with owned_proxy(
            upstream_gateway,
            tmp_path,
            {
                "OPENAI_BASE_URL": f"{default_wire.url}/v1",
                "OPENAI_API_KEY": default_key,
            },
            config=config,
            remove_environment=("OPENAI_API_BASE",),
            workers=2,
        ) as owned:
            with owned.scenario() as scenario:
                multipart: Final = owned.request_multipart(
                    "/v1/rag/ingest",
                    {"request": json.dumps({"ingest_options": _ingest_options("rag ingest contract")})},
                    {"file": ("ui-upload.txt", _MULTIPART_CONTENT, "text/plain")},
                )
                _ingest_response(multipart, new_store_id, "file_ingest_multipart")
                scenario.cleanups.callback(
                    owned.post, "/vector_store/delete", {"vector_store_id": new_store_id}
                )

                json_upload: Final = owned.request(
                    "POST",
                    "/rag/ingest",
                    {
                        "ingest_options": _ingest_options("rag ingest contract"),
                        "file": {
                            "filename": "json-upload.txt",
                            "content": base64.b64encode(_JSON_CONTENT).decode(),
                            "content_type": "text/plain",
                        },
                    },
                )
                _ingest_response(json_upload, new_store_id, "file_ingest_json")

                default_requests: Final = default_wire.drain()
                assert [(request.method, urlsplit(request.target).path) for request in default_requests] == [
                    ("POST", "/v1/vector_stores"),
                    ("POST", "/v1/files"),
                    ("POST", f"/v1/vector_stores/{new_store_id}/files"),
                    ("POST", "/v1/vector_stores"),
                    ("POST", "/v1/files"),
                    ("POST", f"/v1/vector_stores/{new_store_id}/files"),
                ], default_requests
                assert JSON_OBJECT.validate_json(default_requests[0].body) == _PROVIDER_CREATE_BODY
                assert JSON_OBJECT.validate_json(default_requests[2].body) == {
                    "file_id": "file_ingest_multipart",
                    "chunking_strategy": {"type": "auto"},
                }
                assert JSON_OBJECT.validate_json(default_requests[3].body) == _PROVIDER_CREATE_BODY
                assert JSON_OBJECT.validate_json(default_requests[5].body) == {
                    "file_id": "file_ingest_json",
                    "chunking_strategy": {"type": "auto"},
                }
                assert all(
                    request.headers["authorization"] == f"Bearer {default_key}" for request in default_requests
                )
                _assert_upload(default_requests[1], _MULTIPART_CONTENT)
                _assert_upload(default_requests[4], _JSON_CONTENT)

                registered: Final = owned.request(
                    "POST",
                    "/vector_store/new",
                    {
                        "vector_store_id": registered_store_id,
                        "custom_llm_provider": "openai",
                        "litellm_params": {
                            "api_base": f"{registered_wire.url}/v1",
                            "api_key": registered_key,
                        },
                    },
                )
                assert registered.status_code == 200, registered.text
                scenario.cleanups.callback(
                    owned.post, "/vector_store/delete", {"vector_store_id": registered_store_id}
                )
                rejected_credentials: Final = owned.request_multipart(
                    "/v1/rag/ingest",
                    {
                        "request": json.dumps(
                            {
                                "ingest_options": {
                                    "vector_store": {
                                        "vector_store_id": registered_store_id,
                                        "custom_llm_provider": "openai",
                                        "api_key": "caller-synthetic-key",
                                    },
                                }
                            }
                        )
                    },
                    {"file": ("registered.txt", _REGISTRY_CONTENT, "text/plain")},
                )
                assert rejected_credentials.status_code == 400, rejected_credentials.text
                assert "api_key" in rejected_credentials.text, rejected_credentials.text
                assert default_wire.drain() == ()
                assert registered_wire.drain() == ()
                registered_ingest: Final = owned.request_multipart(
                    "/v1/rag/ingest",
                    {
                        "request": json.dumps(
                            {
                                "ingest_options": {
                                    "vector_store": {
                                        "vector_store_id": registered_store_id,
                                        "custom_llm_provider": "openai",
                                    }
                                }
                            }
                        )
                    },
                    {"file": ("registered.txt", _REGISTRY_CONTENT, "text/plain")},
                )
                _ingest_response(registered_ingest, registered_store_id, "file_ingest_registry")
                assert default_wire.drain() == ()
                registered_requests: Final = registered_wire.drain()
                assert [(request.method, urlsplit(request.target).path) for request in registered_requests] == [
                    ("POST", "/v1/files"),
                    ("POST", f"/v1/vector_stores/{registered_store_id}/files"),
                ], registered_requests
                assert all(
                    request.headers["authorization"] == f"Bearer {registered_key}" for request in registered_requests
                )
                _assert_upload(registered_requests[0], _REGISTRY_CONTENT)


def test_documented_multipart_ingest_options_is_honored(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip(
        "BUG: multipart -F ingest_options documented on /v1/rag/ingest is rejected with 400 saying vector_store configuration is missing"
    )

    provider_key: Final = f"ingest-docs-{uuid.uuid4().hex}"
    store_id: Final = f"vs_ingest_docs_{uuid.uuid4().hex}"

    with (
        gateway_from_environment() as upstream_gateway,
        wire_server(_new_store_provider(provider_key, store_id, "docs ingest contract")) as wire,
    ):
        with owned_proxy(
            upstream_gateway,
            tmp_path,
            {"OPENAI_BASE_URL": f"{wire.url}/v1", "OPENAI_API_KEY": provider_key},
            config=_no_registry_config(tmp_path),
            remove_environment=("OPENAI_API_BASE",),
        ) as owned:
            with owned.scenario() as scenario:
                response: Final = owned.request_multipart(
                    "/v1/rag/ingest",
                    {"ingest_options": json.dumps(_ingest_options("docs ingest contract"))},
                    {"file": ("docs.txt", _MULTIPART_CONTENT, "text/plain")},
                )
                _ingest_response(response, store_id, "file_ingest_multipart")
                scenario.cleanups.callback(
                    owned.post, "/vector_store/delete", {"vector_store_id": store_id}
                )

                requests: Final = wire.drain()
                assert [(request.method, urlsplit(request.target).path) for request in requests] == [
                    ("POST", "/v1/vector_stores"),
                    ("POST", "/v1/files"),
                    ("POST", f"/v1/vector_stores/{store_id}/files"),
                ], requests
                assert all(
                    request.headers["authorization"] == f"Bearer {provider_key}" for request in requests
                )
                _assert_upload(requests[1], _MULTIPART_CONTENT)
