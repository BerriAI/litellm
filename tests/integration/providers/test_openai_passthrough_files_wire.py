import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_PROXY_CONFIG: Final = (
    "model_list: []\n"
    "general_settings:\n"
    "  master_key: os.environ/LITELLM_MASTER_KEY\n"
    "  database_url: os.environ/DATABASE_URL\n"
    "  store_model_in_db: true\n"
    "  disable_spend_logs: false\n"
    "  proxy_batch_write_at: 1\n"
)


def test_openai_passthrough_file_upload_and_delete(gateway: Gateway, tmp_path) -> None:
    marker: Final = "openai-file-" + uuid.uuid4().hex
    seen: Final = []

    def respond(request: Request) -> Reply:
        seen.append((request.method, request.target))
        if request.method == "POST" and request.target.endswith("/files"):
            content_type: Final = request.headers.get("content-type", "")
            assert "multipart/form-data" in content_type, request.headers
            body: Final = request.body
            assert b'name="purpose"' in body and b"assistants" in body, body[:400]
            assert b'name="file"' in body and marker.encode() in body, body[:400]
            return Reply(
                body=(
                    b'{"id": "file-' + marker.encode() + b'", "object": "file", "bytes": 12, '
                    b'"created_at": 1700000000, "purpose": "assistants", "filename": "notes.txt"}'
                ),
            )
        if request.method == "DELETE" and request.target.endswith(f"/files/file-{marker}"):
            return Reply(body=b'{"id": "file-' + marker.encode() + b'", "object": "file", "deleted": true}')
        return Reply(status=404)

    with wire_server(respond) as wire:
        config: Final = tmp_path / "proxy_config.yaml"
        config.write_text(
            _PROXY_CONFIG
            + "files_settings:\n"
            + "  - custom_llm_provider: openai\n"
            + f"    api_base: {wire.url}\n"
            + "    api_key: synthetic-openai-key\n"
        )
        with owned_proxy(
            gateway,
            tmp_path,
            {"OPENAI_API_BASE": wire.url, "OPENAI_API_KEY": "synthetic-openai-key"},
            config=config,
        ) as candidate:
            upload: Final = candidate.request_multipart(
                "/openai/v1/files",
                {"purpose": "assistants"},
                {"file": ("notes.txt", f"contents {marker}".encode(), "text/plain")},
            )
            assert upload.status_code == 200, upload.text
            assert upload.json()["id"] == f"file-{marker}"
            delete: Final = candidate.client.delete(
                f"{candidate.client.base_url}/openai/v1/files/file-{marker}",
                headers={"Authorization": f"Bearer {candidate.key}"},
            )
            assert delete.status_code == 200, delete.text
            assert delete.json()["deleted"] is True
        assert seen == [("POST", "/files"), ("DELETE", f"/files/file-{marker}")], seen
