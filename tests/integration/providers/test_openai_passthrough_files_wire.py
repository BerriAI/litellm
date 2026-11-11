import uuid
from pathlib import Path
from typing import Final

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_UPSTREAM_KEY: Final = "synthetic-openai-key"


def test_openai_passthrough_file_upload_and_delete(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "openai-file-" + uuid.uuid4().hex
    file_id: Final = f"file-{marker}"

    def respond(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {_UPSTREAM_KEY}", request.headers
        if request.method == "POST" and request.target == "/files":
            assert request.headers["content-type"].startswith("multipart/form-data"), request.headers
            assert b'name="purpose"\r\n\r\nassistants\r\n' in request.body, request.body[:400]
            assert b'filename="notes.txt"' in request.body and marker.encode() in request.body, request.body[:400]
            return Reply(
                body=(
                    b'{"id": "' + file_id.encode() + b'", "object": "file", "bytes": 12, '
                    b'"created_at": 1700000000, "purpose": "assistants", "filename": "notes.txt"}'
                ),
            )
        if request.method == "DELETE" and request.target == f"/files/{file_id}":
            return Reply(body=b'{"id": "' + file_id.encode() + b'", "object": "file", "deleted": true}')
        return Reply(status=404)

    with wire_server(respond) as wire:
        with owned_proxy(
            gateway,
            tmp_path,
            {"OPENAI_API_BASE": wire.url, "OPENAI_API_KEY": _UPSTREAM_KEY},
        ) as candidate:
            upload: Final = candidate.request_multipart(
                "/openai/files",
                {"purpose": "assistants"},
                {"file": ("notes.txt", f"contents {marker}".encode(), "text/plain")},
            )
            assert upload.status_code == 200, upload.text
            assert upload.json()["id"] == file_id
            delete: Final = candidate.request("DELETE", f"/openai/files/{file_id}")
            assert delete.status_code == 200, delete.text
            assert delete.json()["deleted"] is True
        forwarded: Final = tuple((request.method, request.target) for request in wire.drain())
        assert forwarded == (("POST", "/files"), ("DELETE", f"/files/{file_id}")), forwarded
