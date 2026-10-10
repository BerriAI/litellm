import json
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml
from integration._support.client import Gateway, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_ORIGINAL_KEY: Final = "exa-original-provider-key"
_ROTATED_KEY: Final = "exa-rotated-provider-key"
_TOOL_INFO: Final = {"description": "integration search tool crud"}
_QUERY: Final = "search tool crud query"
_KEY_BY_TARGET: Final = MappingProxyType({"/original/search": _ORIGINAL_KEY, "/rotated/search": _ROTATED_KEY})


def _config_without_periodic_reload(directory: Path) -> Path:
    base: Final = yaml.safe_load(_PROXY_CONFIG.read_text())
    config: Final = {
        **base,
        "general_settings": {**base["general_settings"], "proxy_config_reload_interval_seconds": 3600},
    }
    path: Final = directory / "search-tool-crud.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _stored_row(identity: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT search_tool_name, litellm_params, search_tool_info FROM "LiteLLM_SearchToolsTable" '
        "WHERE search_tool_id = %s",
        (identity,),
    )
    assert len(rows) == 1, rows
    return rows[0]


def _assert_stored_encrypted(identity: str, name: str, plaintext: dict[str, str]) -> None:
    row: Final = _stored_row(identity)
    assert (row["search_tool_name"], row["search_tool_info"]) == (name, _TOOL_INFO), row
    stored: Final = row["litellm_params"]
    assert isinstance(stored, dict) and sorted(stored) == sorted(plaintext), row
    leaked: Final = {key: value for key, value in plaintext.items() if stored[key] == value or value in json.dumps(row)}
    assert leaked == {}, f"search tool litellm_params stored in plaintext: {sorted(leaked)}"


def _masked(key: str) -> str:
    return f"{key[:2]}****{key[-2:]}"


def test_search_tool_create_update_delete_is_read_back_and_applied_without_restart(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target in _KEY_BY_TARGET, request.target
        target: Final = request.target
        key: Final = _KEY_BY_TARGET[target]
        assert request.headers["x-api-key"] == key, request.headers
        assert json.loads(request.body) == {"query": _QUERY, "contents": {"text": True}}, request.body
        return Reply(
            body=json.dumps({"results": [{"title": target, "url": "https://crud.invalid", "text": key}]}).encode()
        )

    name: Final = f"crud-tool-{uuid.uuid4().hex}"
    with (
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_config_without_periodic_reload(tmp_path)) as candidate,
        candidate.scenario() as scenario,
    ):
        original: Final = {"search_provider": "exa_ai", "api_key": _ORIGINAL_KEY, "api_base": f"{wire.url}/original"}
        created: Final = candidate.post(
            "/search_tools",
            {"search_tool": {"search_tool_name": name, "litellm_params": original, "search_tool_info": _TOOL_INFO}},
        )
        identity: Final = string_value(created["search_tool_id"])
        scenario.cleanups.callback(candidate.request, "DELETE", f"/search_tools/{identity}")
        _assert_stored_encrypted(identity, name, original)

        def read_back(params: dict[str, str]) -> None:
            expected: Final = {
                "search_tool_id": identity,
                "search_tool_name": name,
                "litellm_params": {**params, "api_key": _masked(params["api_key"])},
                "search_tool_info": _TOOL_INFO,
                "is_from_config": False,
            }
            response: Final = candidate.request("GET", f"/search_tools/{identity}")
            assert response.status_code == 200, response.text
            single: Final = {
                field: value for field, value in response.json().items() if field not in ("created_at", "updated_at")
            }
            assert single == expected, response.text
            listed: Final = [
                {field: value for field, value in tool.items() if field not in ("created_at", "updated_at")}
                for tool in candidate.get("/search_tools/list")["search_tools"]
                if isinstance(tool, dict) and tool["search_tool_id"] == identity
            ]
            assert listed == [expected], listed

        def search_reaches(target: str, key: str) -> None:
            response: Final = candidate.request("POST", f"/v1/search/{name}", {"query": _QUERY})
            assert response.status_code == 200, response.text
            assert response.json() == {
                "object": "search",
                "results": [
                    {"title": target, "url": "https://crud.invalid", "snippet": key, "date": None, "last_updated": None}
                ],
            }, response.text

        read_back(original)
        search_reaches("/original/search", _ORIGINAL_KEY)

        rotated: Final = {"search_provider": "exa_ai", "api_key": _ROTATED_KEY, "api_base": f"{wire.url}/rotated"}
        updated: Final = candidate.request(
            "PUT",
            f"/search_tools/{identity}",
            {"search_tool": {"search_tool_name": name, "litellm_params": rotated, "search_tool_info": _TOOL_INFO}},
        )
        assert updated.status_code == 200, updated.text
        _assert_stored_encrypted(identity, name, rotated)
        read_back(rotated)
        search_reaches("/rotated/search", _ROTATED_KEY)

        deleted: Final = candidate.request("DELETE", f"/search_tools/{identity}")
        assert deleted.status_code == 200, deleted.text
        assert (
            read_rows('SELECT search_tool_id FROM "LiteLLM_SearchToolsTable" WHERE search_tool_id = %s', (identity,))
            == []
        )
        missing: Final = candidate.request("GET", f"/search_tools/{identity}")
        assert missing.status_code == 404, missing.text
        refused: Final = candidate.request("POST", f"/v1/search/{name}", {"query": _QUERY})
        refusal: Final = (refused.status_code >= 400, refused.json().get("error", {}).get("message"))
        assert refusal == (True, f"Search tool '{name}' not found in router.search_tools"), refused.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/original/search"),
            ("POST", "/rotated/search"),
        ]
