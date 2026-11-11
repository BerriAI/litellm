"""Shared rig for the `test_dec1_resources_*` contract tests.

One owned proxy per test module talks to one scripted upstream that answers every OpenAI-compatible
resource route the proxy forwards (files, batches, fine-tuning jobs, assistants, threads, vector stores) on
both the OpenAI path shape (`/v1/...`) and the Azure path shape (`/openai/...?api-version=`). Every credential
the proxy can pick has its own synthetic key, so a test tells which setting supplied the credentials by
reading the `authorization` or `api-key` header the upstream received.

An id, filename or JSON body that contains `fault-<status>` makes the upstream answer that status with an
OpenAI-shaped error body, which is how the error-mapping tests script a failure without a second server.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

FILES_SETTINGS_OPENAI_KEY: Final = "sk-dec1-resources-files-settings-openai"
FILES_SETTINGS_AZURE_KEY: Final = "dec1-resources-files-settings-azure"
FINETUNE_SETTINGS_OPENAI_KEY: Final = "sk-dec1-resources-finetune-settings-openai"
FINETUNE_SETTINGS_AZURE_KEY: Final = "dec1-resources-finetune-settings-azure"
AMBIENT_OPENAI_KEY: Final = "sk-dec1-resources-ambient-openai"
AMBIENT_AZURE_KEY: Final = "dec1-resources-ambient-azure"
ASSISTANTS_KEY: Final = "sk-dec1-resources-assistants"
OPENAI_MODEL: Final = "dec1-resources-openai"
OPENAI_MODEL_KEY: Final = "sk-dec1-resources-deployment-openai"
OTHER_MODEL: Final = "dec1-resources-openai-other"
OTHER_MODEL_KEY: Final = "sk-dec1-resources-deployment-other"
AZURE_MODEL: Final = "dec1-resources-azure"
AZURE_MODEL_KEY: Final = "dec1-resources-deployment-azure"
AZURE_API_VERSION: Final = "2024-10-21"
FILE_CONTENT: Final = b"dec1 scripted file content\n"
CREATED_AT: Final = 1700000000
_FAULT: Final = re.compile(r"fault-(\d{3})")
_MODEL_PROBES: Final = frozenset({"/v1/models", "/openai/models", "/models"})
_REWRITTEN_BATCH_STEM: Final = "modified_file"


def batch_line(model: str, custom_id: str = "request-1", content: str = "one") -> dict[str, JsonValue]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": [{"role": "user", "content": content}]},
    }


def jsonl(*lines: Mapping[str, JsonValue]) -> bytes:
    return b"".join(json.dumps(line, separators=(",", ":")).encode() + b"\n" for line in lines)


def unique(stem: str) -> str:
    return f"{stem}-{uuid.uuid4().hex[:12]}"


def bearer(request: Request) -> str:
    """The provider credential a request carried: `Authorization: Bearer` on OpenAI, `api-key` on Azure."""
    return request.headers.get("authorization", request.headers.get("api-key", "")).removeprefix("Bearer ")


def path_of(request: Request) -> str:
    return urlsplit(request.target).path


def query_of(request: Request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.target).query)


def json_body(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def multipart_parts(request: Request) -> tuple[Message, ...]:
    header: Final = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    parsed: Final = BytesParser(policy=HTTP).parsebytes(header + request.body)
    return tuple(parsed.iter_parts())


def multipart_field(request: Request, name: str) -> bytes:
    """The raw bytes of the one multipart part named `name` in an upstream upload."""
    matches: Final = [
        part for part in multipart_parts(request) if part.get_param("name", header="content-disposition") == name
    ]
    assert len(matches) == 1, f"Expected one multipart part named {name!r}: {request.body[:300]!r}"
    payload: Final = matches[0].get_payload(decode=True)
    assert isinstance(payload, bytes)
    return payload


def multipart_filename(request: Request) -> str:
    names: Final = [part.get_filename() for part in multipart_parts(request) if part.get_filename()]
    assert len(names) == 1, request.body[:300]
    return str(names[0])


def model_encoded_id(provider_id: str, model: str, prefix: str = "file-") -> str:
    """The id the proxy hands a caller for a model-routed object: prefix + base64url(`litellm:<id>;model,<model>`)."""
    raw: Final = f"litellm:{provider_id};model,{model}".encode()
    return prefix + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decoded(unified_id: str) -> str:
    """Decode a managed (unified) id, which is unpadded base64url of a `litellm_proxy...` string."""
    return base64.urlsafe_b64decode(unified_id + "=" * (-len(unified_id) % 4)).decode()


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    """The `error` object of an OpenAI-shaped error response."""
    body: Final = JSON_OBJECT.validate_json(response.content)
    return JSON_OBJECT.validate_python(body["error"])


def file_object(
    file_id: str, filename: str = "dec1.jsonl", purpose: str = "batch", size: int = 123
) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "bytes": size,
        "created_at": CREATED_AT,
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
    }


def batch_object(batch_id: str, status: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-" + batch_id.removeprefix("batch_"),
        "status": status,
        "created_at": CREATED_AT,
        "completion_window": "24h",
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
        **fields,
    }


def job_object(job_id: str, status: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": job_id,
        "object": "fine_tuning.job",
        "created_at": CREATED_AT,
        "error": None,
        "fine_tuned_model": None,
        "finished_at": None,
        "hyperparameters": {"n_epochs": 3},
        "model": "gpt-4o-mini-2024-07-18",
        "organization_id": "org-dec1-resources",
        "result_files": [],
        "seed": 42,
        "status": status,
        "trained_tokens": None,
        "training_file": "file-" + job_id.removeprefix("ftjob-"),
        "validation_file": None,
        **fields,
    }


def vector_store_object(store_id: str, name: str = "dec1 store") -> dict[str, JsonValue]:
    return {
        "id": store_id,
        "object": "vector_store",
        "created_at": CREATED_AT,
        "name": name,
        "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
        "status": "completed",
        "usage_bytes": 0,
        "expires_after": None,
        "expires_at": None,
        "last_active_at": None,
        "metadata": {},
    }


def _json(body: Mapping[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def _page(*items: Mapping[str, JsonValue]) -> Reply:
    first: Final = items[0]["id"] if items else None
    last: Final = items[-1]["id"] if items else None
    return _json({"object": "list", "data": list(items), "first_id": first, "last_id": last, "has_more": False})


def _fault(status: int) -> Reply:
    kinds: Final = {400: "invalid_request_error", 404: "invalid_request_error", 429: "rate_limit_error"}
    return _json(
        {
            "error": {
                "message": f"scripted upstream fault {status}",
                "type": kinds.get(status, "server_error"),
                "param": None,
                "code": f"scripted_{status}",
            }
        },
        status,
    )


def _assistant(assistant_id: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": assistant_id,
        "object": "assistant",
        "created_at": CREATED_AT,
        "name": "dec1 assistant",
        "description": None,
        "model": "gpt-4o-mini",
        "instructions": None,
        "tools": [],
        "metadata": {},
        **fields,
    }


def _thread(thread_id: str) -> dict[str, JsonValue]:
    return {"id": thread_id, "object": "thread", "created_at": CREATED_AT, "metadata": {}, "tool_resources": {}}


def _message(thread_id: str) -> dict[str, JsonValue]:
    return {
        "id": "msg_dec1",
        "object": "thread.message",
        "created_at": CREATED_AT,
        "thread_id": thread_id,
        "role": "user",
        "content": [{"type": "text", "text": {"value": "hello", "annotations": []}}],
        "assistant_id": None,
        "run_id": None,
        "attachments": [],
        "metadata": {},
        "status": "completed",
    }


def _upload_stem(filename: str, content: bytes) -> str:
    """The stem the scripted provider names an uploaded file after: the filename without its extension.

    A managed batch input is rewritten by the proxy and re-uploaded as `modified_file.jsonl`
    (litellm/litellm_core_utils/batch_utils.py:206) whatever the caller called it, so for that name the stem is
    the `custom_id` of the file's first request line instead, which the rewrite keeps. Without this every
    managed upload would get the same provider id."""
    stem: Final = filename.rsplit(".", 1)[0]
    if stem != _REWRITTEN_BATCH_STEM:
        return stem
    first_line: Final = content.decode("utf-8").strip().split("\n", 1)[0]
    custom_id: Final = JSON_OBJECT.validate_json(first_line).get("custom_id")
    return custom_id if isinstance(custom_id, str) and custom_id else stem


@dataclass(slots=True)
class Backend:
    """The scripted provider. Remembers uploaded file bytes so a later content download returns them."""

    uploads: dict[str, bytes] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def respond(self, request: Request) -> Reply:
        path: Final = path_of(request).removeprefix("/openai").removeprefix("/v1")
        if path == "/models":
            return _json({"object": "list", "data": []})
        fault: Final = _FAULT.search(request.target + request.body[:4096].decode("latin-1"))
        if fault is not None:
            return _fault(int(fault.group(1)))
        segments: Final = tuple(path.strip("/").split("/"))
        answer: Final = self._resource(request.method, segments, request)
        if answer is None:
            return _json({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)
        return answer

    def _resource(self, method: str, segments: tuple[str, ...], request: Request) -> Reply | None:
        match (method, segments):
            case ("POST", ("files",)):
                return self._upload(request)
            case ("GET", ("files",)):
                return _page(file_object("file-dec1-listed-a"), file_object("file-dec1-listed-b"))
            case ("GET", ("files", file_id, "content")):
                with self.lock:
                    content = self.uploads.get(file_id, FILE_CONTENT)
                return Reply(body=content, content_type="application/octet-stream")
            case ("GET", ("files", file_id)):
                return _json(file_object(file_id))
            case ("DELETE", ("files", file_id)):
                return _json({"id": file_id, "object": "file", "deleted": True})
            case ("POST", ("batches",)):
                sent = json_body(request)
                created = "batch_" + str(sent["input_file_id"]).removeprefix("file-")
                return _json(batch_object(created, "validating", **sent))
            case ("GET", ("batches",)):
                return _page(batch_object("batch_dec1-listed-a", "in_progress"))
            case ("POST", ("batches", batch_id, "cancel")):
                return _json(batch_object(batch_id, "cancelling"))
            case ("GET", ("batches", batch_id)):
                return _json(batch_object(batch_id, "in_progress"))
            case ("POST", ("fine_tuning", "jobs")):
                sent = json_body(request)
                created = "ftjob-" + str(sent["training_file"]).removeprefix("file-")
                return _json(job_object(created, "validating_files", training_file=sent["training_file"]))
            case ("GET", ("fine_tuning", "jobs")):
                return _page(job_object("ftjob-dec1-listed-a", "running"))
            case ("POST", ("fine_tuning", "jobs", job_id, "cancel")):
                return _json(job_object(job_id, "cancelled"))
            case ("GET", ("fine_tuning", "jobs", job_id)):
                return _json(job_object(job_id, "running"))
        return self._assistants(method, segments, request) or self._vector_stores(method, segments, request)

    def _upload(self, request: Request) -> Reply:
        filename: Final = multipart_filename(request)
        content: Final = multipart_field(request, "file")
        purpose: Final = multipart_field(request, "purpose").decode()
        file_id: Final = "file-" + _upload_stem(filename, content)
        with self.lock:
            self.uploads[file_id] = content
        return _json(file_object(file_id, filename, purpose, len(content)))

    @staticmethod
    def _assistants(method: str, segments: tuple[str, ...], request: Request) -> Reply | None:
        match (method, segments):
            case ("GET", ("assistants",)):
                return _page(_assistant("asst_dec1-listed-a"))
            case ("POST", ("assistants",)):
                sent = json_body(request)
                return _json(_assistant("asst_" + str(sent.get("name", "dec1")), **sent))
            case ("DELETE", ("assistants", assistant_id)):
                return _json({"id": assistant_id, "object": "assistant.deleted", "deleted": True})
            case ("POST", ("threads",)):
                return _json(_thread("thread_dec1-created"))
            case ("GET", ("threads", thread_id)):
                return _json(_thread(thread_id))
            case ("POST", ("threads", thread_id, "messages")):
                return _json(_message(thread_id))
            case ("GET", ("threads", thread_id, "messages")):
                return _page(_message(thread_id))
        return None

    @staticmethod
    def _vector_stores(method: str, segments: tuple[str, ...], request: Request) -> Reply | None:
        match (method, segments):
            case ("POST", ("vector_stores",)):
                sent = json_body(request)
                return _json(vector_store_object("vs_" + str(sent.get("name", "dec1")), str(sent.get("name", "dec1"))))
            case ("GET", ("vector_stores",)):
                return _page(vector_store_object("vs_dec1-listed-a"))
            case ("POST", ("vector_stores", store_id, "search")):
                return _json(
                    {
                        "object": "vector_store.search_results.page",
                        "search_query": [str(json_body(request).get("query", ""))],
                        "data": [
                            {
                                "file_id": "file-dec1-hit",
                                "filename": "hit.txt",
                                "score": 0.75,
                                "attributes": {"store": store_id},
                                "content": [{"type": "text", "text": "scripted hit"}],
                            }
                        ],
                        "has_more": False,
                        "next_page": None,
                    }
                )
            case ("GET", ("vector_stores", store_id)):
                return _json(vector_store_object(store_id))
            case ("POST", ("vector_stores", store_id)):
                return _json(vector_store_object(store_id, str(json_body(request).get("name", "dec1 store"))))
            case ("DELETE", ("vector_stores", store_id)):
                return _json({"id": store_id, "object": "vector_store.deleted", "deleted": True})
        return None


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire
    backend: Backend

    def drained(self) -> tuple[Request, ...]:
        """Every upstream request since the last drain, without the proxy's own model-discovery probes."""
        return tuple(request for request in self.wire.drain() if path_of(request) not in _MODEL_PROBES)

    def trail(self) -> list[tuple[str, str]]:
        return [(request.method, path_of(request)) for request in self.drained()]

    def upload(
        self,
        filename: str,
        content: bytes,
        *,
        purpose: str = "batch",
        key: str | None = None,
        content_type: str = "application/jsonl",
        **fields: str,
    ) -> httpx.Response:
        return self.proxy.request_multipart(
            "/v1/files", {"purpose": purpose, **fields}, {"file": (filename, content, content_type)}, key=key
        )

    def uploaded_id(self, filename: str, content: bytes, *, key: str | None = None, **fields: str) -> str:
        response: Final = self.upload(filename, content, key=key, **fields)
        assert response.status_code == 200, response.text
        identity: Final = JSON_OBJECT.validate_json(response.content)["id"]
        assert isinstance(identity, str)
        return identity

    def spend_rows(self, key: str, call_type: str) -> list[dict[str, JsonValue]]:
        """Poll SpendLogs until the virtual key has at least one row of `call_type`; rows land asynchronously."""
        return eventually(
            lambda: self.proxy.rows(
                'SELECT request_id, call_type, spend, status, model, api_base FROM "LiteLLM_SpendLogs" '
                "WHERE api_key = %s AND call_type = %s",
                (sha256(key.encode()).hexdigest(), call_type),
            ),
            lambda rows: len(rows) >= 1,
            seconds=70,
        )


def resource_config(wire_url: str, litellm_settings: Mapping[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    openai_base: Final = f"{wire_url}/v1"
    return {
        "model_list": [
            {
                "model_name": OPENAI_MODEL,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": OPENAI_MODEL_KEY, "api_base": openai_base},
            },
            {
                "model_name": OTHER_MODEL,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": OTHER_MODEL_KEY, "api_base": openai_base},
            },
            {
                "model_name": AZURE_MODEL,
                "litellm_params": {
                    "model": "azure/dec1-resources-deployment",
                    "api_key": AZURE_MODEL_KEY,
                    "api_base": wire_url,
                    "api_version": AZURE_API_VERSION,
                },
            },
        ],
        "files_settings": [
            {"custom_llm_provider": "openai", "api_key": FILES_SETTINGS_OPENAI_KEY, "api_base": openai_base},
            {
                "custom_llm_provider": "azure",
                "api_key": FILES_SETTINGS_AZURE_KEY,
                "api_base": wire_url,
                "api_version": AZURE_API_VERSION,
            },
        ],
        "finetune_settings": [
            {"custom_llm_provider": "openai", "api_key": FINETUNE_SETTINGS_OPENAI_KEY, "api_base": openai_base},
            {
                "custom_llm_provider": "azure",
                "api_key": FINETUNE_SETTINGS_AZURE_KEY,
                "api_base": wire_url,
                "api_version": AZURE_API_VERSION,
            },
        ],
        "assistant_settings": {
            "custom_llm_provider": "openai",
            "litellm_params": {"api_key": ASSISTANTS_KEY, "api_base": openai_base, "max_retries": 0},
        },
        "router_settings": {"num_retries": 0},
        "litellm_settings": dict(litellm_settings or {}),
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "database_url": "os.environ/DATABASE_URL"},
    }


def resource_environment(wire_url: str) -> dict[str, str]:
    """Ambient provider credentials, which the proxy falls back to when no setting names a credential."""
    return {
        "OPENAI_BASE_URL": f"{wire_url}/v1",
        "OPENAI_API_BASE": f"{wire_url}/v1",
        "OPENAI_API_KEY": AMBIENT_OPENAI_KEY,
        "AZURE_API_BASE": wire_url,
        "AZURE_API_KEY": AMBIENT_AZURE_KEY,
        "AZURE_API_VERSION": AZURE_API_VERSION,
    }


@contextmanager
def booted_rig(
    directory: Path,
    *,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    remove_environment: tuple[str, ...] = (),
    drop_settings: tuple[str, ...] = (),
) -> Iterator[Rig]:
    """Boot one owned proxy against one scripted upstream; `drop_settings` removes top-level config sections."""
    backend: Final = Backend()
    with gateway_from_environment() as shared, wire_server(backend.respond) as wire:
        settings: Final = resource_config(wire.url, litellm_settings)
        merged: Final = {
            **{name: value for name, value in settings.items() if name not in drop_settings},
            "general_settings": {
                **JSON_OBJECT.validate_python(settings["general_settings"]),
                **(general_settings or {}),
            },
        }
        config: Final = directory / "dec1-resources-config.yaml"
        config.write_text(yaml.safe_dump(merged))
        environment: Final = {
            name: value for name, value in resource_environment(wire.url).items() if name not in remove_environment
        }
        with owned_proxy(shared, directory, environment, config=config, remove_environment=remove_environment) as proxy:
            yield Rig(proxy, wire, backend)
