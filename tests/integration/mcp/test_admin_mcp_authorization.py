"""Hosted admin MCP authorizes a proxy admin no matter how many keys the proxy holds.

The connector reads the caller's role from `/user/info?admin_mcp=true` and refuses any response over
2,000,000 bytes (liteadmin-mcp 31fb0969, gateway.py `_request`, read 2026-11-16), so the seed below puts
more key metadata than that on the proxy.
"""

import uuid
from pathlib import Path
from typing import Final

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from pydantic import TypeAdapter

_SEEDED_KEYS: Final = 30
_METADATA_BYTES_PER_KEY: Final = 100_000
_TOOL_NAMES: Final = TypeAdapter(dict[str, object])


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
        result: Final = _TOOL_NAMES.validate_python(_TOOL_NAMES.validate_json(listed.content)["result"])
        assert result["tools"], listed.text[:500]
