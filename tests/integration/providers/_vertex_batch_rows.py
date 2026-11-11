import json
import threading
from collections.abc import Callable
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import httpx
from integration._support.client import Gateway, Scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

PROJECT: Final = "scripted-project"
LOCATION: Final = "us-central1"
BUCKET: Final = "scripted-bucket"
UPLOAD_PREFIX: Final = f"/upload/storage/v1/b/{BUCKET}/o?uploadType=media&name="
CHAT_URL: Final = "/v1/chat/completions"
BOM: Final = b"\xef\xbb\xbf"
BODY_NOT_OBJECT: Final = "`body` on batch requests must be a JSON object, but was missing or not an object"
Route = Literal["model", "target_model_names"]
ROUTES: Final[tuple[Route, ...]] = ("model", "target_model_names")
_ERROR_BODY: Final = TypeAdapter(dict[str, dict[str, JsonValue]])


def stored_object(request: Request) -> Reply:
    name: Final = unquote(request.target.removeprefix(UPLOAD_PREFIX))
    return Reply(
        body=json.dumps(
            {
                "kind": "storage#object",
                "id": f"{BUCKET}/{name}/1759950000000000",
                "name": name,
                "bucket": BUCKET,
                "size": str(len(request.body)),
                "timeCreated": "2026-10-15T06:00:00.000Z",
                "contentType": "application/octet-stream",
            }
        ).encode()
    )


def is_gcs_upload(request: Request) -> bool:
    return request.method == "POST" and request.target.startswith(UPLOAD_PREFIX)


def gcs_peer(request: Request) -> Reply:
    assert is_gcs_upload(request), (request.method, request.target)
    assert request.headers["authorization"] == "Bearer scripted-token"
    return stored_object(request)


def held_gcs_peer(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert release.wait(timeout=120), "the held bucket upload was never released"
        return gcs_peer(request)

    return respond


def vertex_params(gateway: Gateway, url: str, backend: str = "gemini-2.5-flash") -> dict[str, JsonValue]:
    return {
        "model": f"vertex_ai/{backend}",
        "api_key": None,
        "api_base": url,
        "vertex_project": PROJECT,
        "vertex_location": LOCATION,
        "vertex_credentials": service_account_json(PROJECT, gateway.upstream_url),
        "gcs_bucket_name": BUCKET,
    }


def vertex_model(gateway: Gateway, scenario: Scenario, url: str, backend: str = "gemini-2.5-flash") -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_key=None,
        api_base=url,
        vertex_project=PROJECT,
        vertex_location=LOCATION,
        vertex_credentials=service_account_json(PROJECT, gateway.upstream_url),
        gcs_bucket_name=BUCKET,
    )


def chat_row(custom_id: str, model: str, text: str) -> dict[str, JsonValue]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT_URL,
        "body": {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 8},
    }


def roleless_row(custom_id: str, model: str) -> dict[str, JsonValue]:
    return {"custom_id": custom_id, "method": "POST", "url": CHAT_URL, "body": {"model": model, "messages": [{}]}}


def encoded(row: dict[str, JsonValue]) -> bytes:
    return json.dumps(row).encode()


def jsonl(*lines: bytes) -> bytes:
    return b"\n".join(lines) + b"\n"


def upload(
    gateway: Gateway, route: Route | None, model: str, content: bytes, *, client: httpx.Client | None = None, **extra: str
) -> httpx.Response:
    fields: Final = {"purpose": "batch", **({route: model} if route is not None else {}), **extra}
    return (client or gateway.client).post(
        "/v1/files",
        data=fields,
        files={"file": ("rows.jsonl", content, "application/jsonl")},
        headers={"Authorization": f"Bearer {gateway.key}"},
    )


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return _ERROR_BODY.validate_json(response.content)["error"]


def unconvertible_line_message(line: int) -> str:
    return f"Batch input file line {line} could not be converted to a Vertex AI request: "


def stored_custom_ids(request: Request) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["request"]["labels"]["litellm_custom_id"] for line in request.body.decode().split("\n") if line
    )


def multipart_file(request: Request) -> bytes:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parts: Final[tuple[Message, ...]] = tuple(BytesParser(policy=HTTP).parsebytes(envelope).iter_parts())
    files: Final = tuple(part.get_payload(decode=True) for part in parts if part.get_filename() is not None)
    assert len(files) == 1, [part.get("content-disposition") for part in parts]
    payload: Final = files[0]
    assert isinstance(payload, bytes)
    return payload


def is_openai_upload(request: Request) -> bool:
    return request.method == "POST" and urlsplit(request.target).path == "/v1/files"


def openai_files_peer(request: Request) -> Reply:
    if not is_openai_upload(request):
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    return Reply(
        body=json.dumps(
            {
                "id": "file-scripted",
                "object": "file",
                "bytes": len(multipart_file(request)),
                "created_at": 1792040000,
                "filename": "modified_file.jsonl",
                "purpose": "batch",
                "status": "uploaded",
            }
        ).encode()
    )


def openai_uploaded_lines(request: Request) -> list[str]:
    return multipart_file(request).decode().split("\n")
