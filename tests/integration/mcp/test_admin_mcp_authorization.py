"""Hosted admin MCP authorizes a proxy admin no matter how many keys the proxy holds.

The connector reads the caller's role from `/user/info?admin_mcp=true` and refuses any response over
2,000,000 bytes (liteadmin-mcp 31fb0969, gateway.py `_request`, read 2026-11-16), so the seed below puts
more key metadata than that on the proxy.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy, owned_proxy_process
from pydantic import TypeAdapter

_SEEDED_KEYS: Final = 30
_METADATA_BYTES_PER_KEY: Final = 100_000
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def test_admin_mcp_lists_tools_when_proxy_key_payload_exceeds_connector_byte_limit(
    gateway: Gateway, tmp_path: Path
) -> None:
    admin_id: Final = f"admin-mcp-{uuid.uuid4().hex}"
    with owned_proxy(gateway, tmp_path, {"LITELLM_ENABLE_ADMIN_MCP": "true"}) as candidate:
        candidate.post("/user/new", {"user_id": admin_id, "user_role": "proxy_admin", "auto_create_key": False})
        admin_key: Final = str(candidate.post("/key/generate", {"user_id": admin_id})["key"])
        for index in range(_SEEDED_KEYS):
            candidate.post(
                "/key/generate",
                {"key_alias": f"{admin_id}-{index}", "metadata": {"note": "x" * _METADATA_BYTES_PER_KEY}},
            )
        every_key: Final = candidate.request("GET", "/user/info", key=admin_key)
        assert every_key.status_code == 200, every_key.text[:500]
        assert len(every_key.content) > _SEEDED_KEYS * _METADATA_BYTES_PER_KEY, len(every_key.content)

        listed: Final = candidate.request(
            "POST",
            "/admin/mcp",
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            key=admin_key,
            headers={"Accept": "application/json, text/event-stream"},
        )

        assert listed.status_code == 200, listed.text[:500]
        result: Final = _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_json(listed.content)["result"])
        assert result["tools"], listed.text[:500]

        own_row: Final = candidate.request("GET", "/user/info", key=admin_key, params={"admin_mcp": "true"})
        assert own_row.status_code == 200, own_row.text[:500]
        compact: Final = _JSON_OBJECT.validate_json(own_row.content)
        assert (compact["user_id"], compact["keys"], compact["teams"]) == (admin_id, [], []), own_row.text[:500]


def _list_tools(candidate: Gateway, key: str | None) -> tuple[int, str]:
    listed: Final = candidate.request(
        "POST",
        "/admin/mcp",
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        key=key,
        headers={"Accept": "application/json, text/event-stream"},
    )
    return listed.status_code, listed.text[:200]


def _key_and_team_counts(candidate: Gateway, key: str | None, params: dict[str, str]) -> tuple[int, int, int]:
    info: Final = candidate.request("GET", "/user/info", key=key, params=params)
    body: Final = _JSON_OBJECT.validate_json(info.content) if info.status_code == 200 else {"keys": [], "teams": []}
    keys: Final = TypeAdapter(list[object]).validate_python(body["keys"])
    teams: Final = TypeAdapter(list[object]).validate_python(body["teams"])
    return info.status_code, len(keys), len(teams)


def test_admin_mcp_user_info_stays_scoped_per_role_and_survives_a_worker_kill_over_the_connector_byte_limit(
    gateway: Gateway, tmp_path: Path
) -> None:
    suffix: Final = uuid.uuid4().hex
    admin_id: Final = f"admin-mcp-{suffix}"
    viewer_id: Final = f"viewer-mcp-{suffix}"
    member_id: Final = f"member-mcp-{suffix}"
    flag: Final = {"admin_mcp": "true"}
    with owned_proxy_process(gateway, tmp_path, {"LITELLM_ENABLE_ADMIN_MCP": "true"}, workers=2) as owned:
        candidate: Final = owned.gateway
        candidate.post("/user/new", {"user_id": admin_id, "user_role": "proxy_admin", "auto_create_key": False})
        candidate.post("/user/new", {"user_id": viewer_id, "user_role": "proxy_admin_viewer", "auto_create_key": False})
        candidate.post("/user/new", {"user_id": member_id, "user_role": "internal_user", "auto_create_key": False})
        admin_key: Final = str(candidate.post("/key/generate", {"user_id": admin_id})["key"])
        viewer_key: Final = str(candidate.post("/key/generate", {"user_id": viewer_id})["key"])
        member_key: Final = str(candidate.post("/key/generate", {"user_id": member_id})["key"])
        candidate.post("/team/new", {"team_alias": f"team-mcp-{suffix}"})
        for index in range(_SEEDED_KEYS):
            candidate.post(
                "/key/generate",
                {"key_alias": f"{admin_id}-{index}", "metadata": {"note": "x" * _METADATA_BYTES_PER_KEY}},
            )
        every_key: Final = _SEEDED_KEYS + 3

        assert _key_and_team_counts(candidate, admin_key, {}) == (200, every_key, 1)
        assert _key_and_team_counts(candidate, admin_key, flag) == (200, 0, 0)
        assert _key_and_team_counts(candidate, admin_key, {**flag, "mcp_tool": "list_keys"}) == (200, 0, 0)
        assert _key_and_team_counts(candidate, admin_key, {**flag, "user_id": member_id}) == (200, 1, 0)
        assert _key_and_team_counts(candidate, admin_key, {"admin_mcp": "false"}) == (200, every_key, 1)
        assert _key_and_team_counts(candidate, admin_key, {"admin_mcp": "x" * 5000})[0] == 422
        assert _key_and_team_counts(candidate, viewer_key, flag) == (200, 0, 0)
        assert _key_and_team_counts(candidate, None, flag) == (200, 0, 0)
        assert _key_and_team_counts(candidate, member_key, flag) == (200, 1, 0)
        assert _key_and_team_counts(candidate, member_key, {}) == (200, 1, 0)
        assert _key_and_team_counts(candidate, "sk-not-a-key", flag)[0] == 401

        assert [_list_tools(candidate, admin_key)[0] for _ in range(20)] == [200] * 20
        assert _list_tools(candidate, viewer_key)[0] == 403
        assert _list_tools(candidate, None)[0] == 403
        assert _list_tools(candidate, member_key)[0] == 403

        workers: Final = eventually(owned.worker_pids, lambda pids: len(pids) >= 2, seconds=30)
        owned.kill_worker(min(workers))
        with ThreadPoolExecutor(max_workers=8) as pool:
            after_kill: Final = tuple(pool.map(lambda _: _list_tools(candidate, admin_key)[0], range(20)))
        assert after_kill == (200,) * 20
