from __future__ import annotations

import json
from typing import Final
from urllib.parse import unquote

import httpx
from integration._support.client import Gateway, Scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

BACKEND: Final = "gemini-embedding-2-preview"
PROJECT: Final = "scripted-project"
LOCATION: Final = "us-central1"
BUCKET: Final = "scripted-bucket"
UPLOAD_PREFIX: Final = f"/upload/storage/v1/b/{BUCKET}/o?uploadType=media&name="
GCS_URI: Final = "gs://scripted-bucket/clips/animals.mp4"
BLOCK: Final[dict[str, JsonValue]] = {
    "type": "file",
    "file": {"file_id": GCS_URI, "video_metadata": {"fps": 1.0, "start_offset": "0s", "end_offset": "3s"}},
}
GCS_PART: Final[dict[str, JsonValue]] = {
    "file_data": {"mime_type": "video/mp4", "file_uri": GCS_URI},
    "video_metadata": {"fps": 1.0, "startOffset": "0s", "endOffset": "3s"},
}
TEXT_PART: Final[dict[str, JsonValue]] = {"text": "a red bus"}


def row(custom_id: str, model: str, elements: JsonValue) -> dict[str, JsonValue]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": model, "input": elements},
    }


def stored_row(key: str, parts: list[JsonValue]) -> dict[str, JsonValue]:
    return {"key": key, "request": {"content": {"parts": parts}}}


def gcs_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target.startswith(UPLOAD_PREFIX), request.target
    assert request.headers["authorization"] == "Bearer scripted-token"
    name: Final = unquote(request.target.removeprefix(UPLOAD_PREFIX))
    stored: Final = {
        "kind": "storage#object",
        "id": f"{BUCKET}/{name}/1759950000000000",
        "name": name,
        "bucket": BUCKET,
        "size": str(len(request.body)),
        "timeCreated": "2026-10-08T19:00:00.000Z",
        "contentType": "application/octet-stream",
    }
    return Reply(body=json.dumps(stored).encode())


def vertex_batch_model(gateway: Gateway, scenario: Scenario, url: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{BACKEND}",
        api_key=None,
        api_base=url,
        vertex_project=PROJECT,
        vertex_location=LOCATION,
        vertex_credentials=service_account_json(PROJECT, gateway.upstream_url),
        gcs_bucket_name=BUCKET,
    )


def upload(gateway: Gateway, model: str, rows: tuple[dict[str, JsonValue], ...]) -> httpx.Response:
    jsonl: Final = "\n".join(json.dumps(line) for line in rows) + "\n"
    return gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model},
        {"file": ("in.jsonl", jsonl.encode(), "application/jsonl")},
    )


def test_block_rows_upload_as_embed_content_requests(gateway: Gateway) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_batch_model(gateway, scenario, wire.url)
        rows: Final = (
            row("mixed", model, [BLOCK, "a red bus"]),
            row("bare", model, "a red bus"),
            row("nested", model, [[BLOCK, "a red bus"]]),
        )
        response: Final = upload(gateway, model, rows)
        assert response.status_code == 200, response.text
        assert response.json()["object"] == "file" and response.json()["purpose"] == "batch", response.text
        uploads: Final = wire.drain()
        assert len(uploads) == 1, [upload.target for upload in uploads]
        stored: Final = tuple(json.loads(line) for line in uploads[0].body.decode().splitlines() if line.strip())
        assert stored == (
            stored_row("mixed#0/2", [GCS_PART]),
            stored_row("mixed#1/2", [TEXT_PART]),
            stored_row("bare", [TEXT_PART]),
            stored_row("nested", [GCS_PART, TEXT_PART]),
        )


def test_bare_block_row_fails_the_upload_before_any_storage_write(gateway: Gateway) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_batch_model(gateway, scenario, wire.url)
        response: Final = upload(gateway, model, (row("mixed", model, [BLOCK, "a red bus"]), row("bare", model, BLOCK)))
        assert response.status_code >= 400, response.text
        assert "got dict" in response.text, response.text
        assert wire.drain() == (), "a rejected batch file reached the bucket"
