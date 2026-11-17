"""Shared helpers for the `test_dec1_batchfiles_*` contract tests (files and batches on the big providers).

Every test drives the proxy black-box against a scripted upstream started with `wire_server`. The helpers
here build the provider-shaped objects each upstream answers with (a Gemini batch operation, an Anthropic
Message Batch, a Vertex batch prediction job, a Bedrock model invocation job, an OpenAI batch output line),
decode the ids the proxy hands back, and read the spend rows a completed batch writes.

Id formats pinned by these tests (read from `litellm/proxy/openai_files_endpoints/common_utils.py`
`encode_file_id_with_model` and `litellm/types/utils.py` `SpecialEnums`):
a model-routed id is `<prefix>` + unpadded base64url of `litellm:<provider id>;model,<model name>` where the
prefix is `file-` for files and `batch_` for batches; a managed batch id is unpadded base64url of
`litellm_proxy;model_id:<deployment id>;llm_batch_id:<provider batch id>`.
"""

from __future__ import annotations

import base64
import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from hashlib import sha256
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
CHAT_URL: Final = "/v1/chat/completions"
GEMINI_KEY: Final = "dec1-batchfiles-gemini-key"
ANTHROPIC_KEY: Final = "sk-ant-dec1-batchfiles"
AWS_ACCESS_KEY: Final = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY: Final = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
AWS_REGION: Final = "us-east-1"
BEDROCK_MODEL_ID: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_ROLE_ARN: Final = "arn:aws:iam::123456789012:role/dec1-batchfiles-role"
BEDROCK_JOB_ARN_PREFIX: Final = f"arn:aws:bedrock:{AWS_REGION}:123456789012:model-invocation-job/"
S3_BUCKET: Final = "dec1-batchfiles-bucket"
VERTEX_PROJECT: Final = "dec1-batchfiles-project"
VERTEX_LOCATION: Final = "us-central1"
GCS_BUCKET: Final = "dec1-batchfiles-gcs"
CREATED_ISO: Final = "2026-11-17T10:00:00Z"
CREATED_UNIX: Final = 1763373600
ENDED_ISO: Final = "2026-11-17T11:00:00Z"
ENDED_UNIX: Final = 1763377200


def unique(stem: str) -> str:
    return f"{stem}-{uuid.uuid4().hex[:10]}"


def json_of(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def path_of(request: Request) -> str:
    return unquote(urlsplit(request.target).path)


def query_of(request: Request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.target).query)


def model_encoded_id(provider_id: str, model: str, prefix: str) -> str:
    """The id the proxy returns for an object reached through a model: prefix + base64url(`litellm:<id>;model,<m>`)."""
    raw: Final = f"litellm:{provider_id};model,{model}".encode()
    return prefix + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decoded(unified_id: str) -> str:
    """Decode a managed (unified) id: unpadded base64url of a `litellm_proxy...` string."""
    return base64.urlsafe_b64decode(unified_id + "=" * (-len(unified_id) % 4)).decode()


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def chat_row(custom_id: str, model: str, text: str = "dec1 batch row") -> dict[str, JsonValue]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT_URL,
        "body": {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 16},
    }


def jsonl(*rows: Mapping[str, JsonValue]) -> bytes:
    return b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)


def multipart_file_bytes(request: Request) -> bytes:
    """The bytes of the one file part in a multipart upload the upstream received."""
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parts: Final[tuple[Message, ...]] = tuple(BytesParser(policy=HTTP).parsebytes(envelope).iter_parts())
    payloads: Final = [part.get_payload(decode=True) for part in parts if part.get_filename() is not None]
    assert len(payloads) == 1, [part.get("content-disposition") for part in parts]
    payload: Final = payloads[0]
    assert isinstance(payload, bytes)
    return payload


def multipart_filename(request: Request) -> str:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    names: Final = [
        part.get_filename()
        for part in BytesParser(policy=HTTP).parsebytes(envelope).iter_parts()
        if part.get_filename()
    ]
    assert len(names) == 1, names
    return str(names[0])


def reply(body: Mapping[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def text_reply(body: str, status: int = 200) -> Reply:
    return Reply(status=status, body=body.encode(), content_type="application/octet-stream")


Responder = Callable[[Request], Reply]


@dataclass(slots=True)
class Script:
    """A scripted upstream keyed by `(METHOD, path)`; a route may be replaced while the proxy runs.

    Unscripted requests answer 404 with a JSON body naming the request, so a wrong path shows up in the
    proxy's error text instead of as a hang."""

    routes: dict[tuple[str, str], Responder] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def on(self, method: str, path: str, answer: Reply | Responder) -> None:
        responder: Final[Responder] = answer if callable(answer) else (lambda _request, fixed=answer: fixed)
        with self.lock:
            self.routes[(method, path)] = responder

    def __call__(self, request: Request) -> Reply:
        with self.lock:
            responder = self.routes.get((request.method, path_of(request)))
        if responder is None:
            return reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)
        return responder(request)


def sent(requests: tuple[Request, ...], method: str, path: str) -> list[Request]:
    return [request for request in requests if request.method == method and path_of(request) == path]


def openai_output_line(
    custom_id: str, model: str, prompt_tokens: int, completion_tokens: int, *, status: int = 200
) -> dict[str, JsonValue]:
    """One OpenAI batch output line; a non-200 line carries an error body and no usage."""
    if status != 200:
        return {
            "id": f"batch_req_{custom_id}",
            "custom_id": custom_id,
            "response": {"status_code": status, "request_id": custom_id, "body": {"error": {"message": "rejected"}}},
            "error": {"code": "bad_request", "message": "rejected"},
        }
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": custom_id,
            "body": {
                "id": f"chatcmpl-{custom_id}",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        },
        "error": None,
    }


def openai_batch(batch_id: str, status: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": batch_id,
        "object": "batch",
        "endpoint": CHAT_URL,
        "errors": None,
        "input_file_id": f"file-in-{batch_id}",
        "completion_window": "24h",
        "status": status,
        "output_file_id": None,
        "error_file_id": None,
        "created_at": CREATED_UNIX,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
        "metadata": None,
        **fields,
    }


def openai_file(
    file_id: str, purpose: str = "batch", filename: str = "in.jsonl", size: int = 100
) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "bytes": size,
        "created_at": CREATED_UNIX,
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
    }


def gemini_operation(
    batch_id: str,
    state: str,
    *,
    input_file: str = "files/dec1-input",
    responses_file: str | None = None,
    end_time: str | None = None,
    stats: Mapping[str, int] | None = None,
    error: Mapping[str, JsonValue] | None = None,
    display_name: str = "dec1-display",
) -> dict[str, JsonValue]:
    """A Gemini `batches/<id>` long-running operation as `batchGenerateContent` and `GET batches/<id>` return it."""
    metadata: dict[str, JsonValue] = {
        "@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch",
        "model": "models/gemini-2.5-flash",
        "displayName": display_name,
        "inputConfig": {"fileName": input_file},
        "createTime": CREATED_ISO,
        "state": state,
        "batchStats": dict(stats or {}),
    }
    if responses_file is not None:
        metadata["output"] = {"responsesFile": responses_file}
    if end_time is not None:
        metadata["endTime"] = end_time
    operation: dict[str, JsonValue] = {"name": f"batches/{batch_id}", "metadata": metadata}
    if error is not None:
        operation["error"] = dict(error)
    return operation


def anthropic_message_batch(
    batch_id: str,
    processing_status: str,
    *,
    counts: Mapping[str, int] | None = None,
    ended_at: str | None = None,
    cancel_initiated_at: str | None = None,
) -> dict[str, JsonValue]:
    return {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": processing_status,
        "request_counts": {
            "processing": 0,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
            **(counts or {}),
        },
        "ended_at": ended_at,
        "created_at": CREATED_ISO,
        "expires_at": "2026-11-18T10:00:00Z",
        "archived_at": None,
        "cancel_initiated_at": cancel_initiated_at,
        "results_url": None,
    }


def vertex_job(
    job_id: str, state: str, *, input_uri: str, output_directory: str | None = None, model: str = "gemini-2.5-flash"
) -> dict[str, JsonValue]:
    job: dict[str, JsonValue] = {
        "name": f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/batchPredictionJobs/{job_id}",
        "displayName": f"litellm-vertex-batch-{job_id}",
        "model": f"publishers/google/models/{model}",
        "inputConfig": {"gcsSource": {"uris": [input_uri]}, "instancesFormat": "jsonl"},
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": input_uri.rsplit("/", 1)[0]},
        },
        "state": state,
        "createTime": "2026-11-17T10:00:00.000000Z",
        "updateTime": "2026-11-17T10:00:00.000000Z",
    }
    if output_directory is not None:
        job["outputInfo"] = {"gcsOutputDirectory": output_directory}
    return job


def bedrock_job(job_arn: str, status: str, *, input_uri: str, output_uri: str) -> dict[str, JsonValue]:
    return {
        "jobArn": job_arn,
        "jobName": job_arn.rsplit("/", 1)[-1],
        "modelId": BEDROCK_MODEL_ID,
        "roleArn": BEDROCK_ROLE_ARN,
        "status": status,
        "submitTime": "2026-11-17T10:00:00Z",
        "lastModifiedTime": "2026-11-17T10:05:00Z",
        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": input_uri}},
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_uri}},
        "timeoutDurationInHours": 24,
    }


def batch_spend_rows(gateway: Gateway, key: str, *, seconds: float = 70) -> list[dict[str, JsonValue]]:
    """Poll SpendLogs until the key has its completed-batch row (`call_type = 'aretrieve_batch'`)."""
    return eventually(
        lambda: gateway.rows(
            "SELECT request_id, spend, prompt_tokens, completion_tokens, total_tokens, model, metadata, status "
            'FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
            (hashed(key), "aretrieve_batch"),
        ),
        lambda rows: len(rows) >= 1,
        seconds=seconds,
    )


def metadata_of(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    value: Final = row["metadata"]
    return JSON_OBJECT.validate_json(value) if isinstance(value, str) else JSON_OBJECT.validate_python(value)
