import json
from typing import Final

from integration._support.client import Gateway
from integration._support.openai_wire import MODEL_DISCOVERY, answering_model_discovery
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-4o-mini"
_API_KEY: Final = "synthetic-openai-key"
_CONTAINER: Final = "cntr_wire"
_FILE: Final = "cfile_1"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_FILE_OBJECT: Final[dict[str, JsonValue]] = {
    "id": _FILE,
    "object": "container.file",
    "container_id": _CONTAINER,
    "created_at": 1,
    "bytes": 5,
    "path": "/mnt/data/notes.txt",
    "source": "user",
}
_REPLIES: Final = {
    ("POST", "/v1/containers"): Reply(
        body=json.dumps(
            {"id": _CONTAINER, "object": "container", "created_at": 1, "status": "running", "name": "wire"}
        ).encode()
    ),
    ("GET", f"/v1/containers/{_CONTAINER}/files"): Reply(
        body=json.dumps(
            {"object": "list", "data": [_FILE_OBJECT], "first_id": _FILE, "last_id": _FILE, "has_more": False}
        ).encode()
    ),
    ("GET", f"/v1/containers/{_CONTAINER}/files/{_FILE}"): Reply(body=json.dumps(_FILE_OBJECT).encode()),
    ("GET", f"/v1/containers/{_CONTAINER}/files/{_FILE}/content"): Reply(
        body=b"hello", content_type="application/octet-stream"
    ),
    ("DELETE", f"/v1/containers/{_CONTAINER}/files/{_FILE}"): Reply(
        body=json.dumps({"id": _FILE, "object": "container.file.deleted", "deleted": True}).encode()
    ),
}


def _peer(request: Request) -> Reply:
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    return _REPLIES[(request.method, request.target.split("?")[0])]


def test_container_file_routes_reach_the_openai_paths_the_packaged_endpoint_table_declares(
    gateway: Gateway,
) -> None:
    with wire_server(answering_model_discovery(_peer)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        created: Final = gateway.request("POST", "/v1/containers", {"model": model, "name": "wire"})
        assert created.status_code == 200, created.text
        container: Final = str(_JSON_OBJECT.validate_json(created.content)["id"])
        assert container.startswith("cntr_") and container != _CONTAINER, created.text
        files: Final = f"/v1/containers/{container}/files"

        listed: Final = gateway.request("GET", files)
        assert listed.status_code == 200, listed.text
        assert _JSON_OBJECT.validate_json(listed.content)["data"][0]["id"] == _FILE, listed.text

        paged: Final = gateway.request("GET", files, params={"limit": "2", "order": "desc", "after": "cfile_0"})
        assert paged.status_code == 200, paged.text

        retrieved: Final = gateway.request("GET", f"{files}/{_FILE}")
        assert retrieved.status_code == 200, retrieved.text
        assert _JSON_OBJECT.validate_json(retrieved.content)["path"] == "/mnt/data/notes.txt", retrieved.text

        content: Final = gateway.request("GET", f"{files}/{_FILE}/content")
        assert content.status_code == 200 and content.content == b"hello", content.text

        deleted: Final = gateway.request("DELETE", f"{files}/{_FILE}")
        assert deleted.status_code == 200, deleted.text
        assert _JSON_OBJECT.validate_json(deleted.content) == {
            "id": _FILE,
            "object": "container.file.deleted",
            "deleted": True,
        }, deleted.text

        upstream_files: Final = f"/v1/containers/{_CONTAINER}/files"
        calls: Final = [(request.method, request.target) for request in wire.drain()]
        assert [call for call in calls if call != MODEL_DISCOVERY] == [
            ("POST", "/v1/containers"),
            ("GET", upstream_files),
            ("GET", f"{upstream_files}?after=cfile_0&limit=2&order=desc"),
            ("GET", f"{upstream_files}/{_FILE}"),
            ("GET", f"{upstream_files}/{_FILE}/content"),
            ("DELETE", f"{upstream_files}/{_FILE}"),
        ]
