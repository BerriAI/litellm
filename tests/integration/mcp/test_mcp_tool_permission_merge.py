import uuid
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.mcp import (
    call_tool,
    mcp_peer,
    register_mcp,
    tool_names,
)


def test_tool_permissions_merge_when_keys_resolve_to_same_server(gateway: Gateway) -> None:
    with mcp_peer() as first, mcp_peer() as second, gateway.scenario() as scenario:
        shared_alias: Final = "merge" + uuid.uuid4().hex[:8]
        other_alias: Final = "other" + uuid.uuid4().hex[:8]
        first_id: Final = register_mcp(scenario, first, shared_alias)
        second_id: Final = register_mcp(scenario, second, other_alias)
        key: Final = scenario.key(
            object_permission={
                "mcp_servers": [first_id, second_id],
                "mcp_tool_permissions": {
                    shared_alias: ["add"],
                    first_id: ["multiply", "add"],
                    second_id: ["add"],
                },
            }
        )

        first_names: Final = eventually(
            lambda: tool_names(gateway, key, first_id),
            lambda names: set(names) != set(),
            seconds=15,
        )
        assert set(first_names) == {"add", "multiply"}, first_names
        assert set(tool_names(gateway, key, second_id)) == {"add"}

        first.drain()
        add: Final = call_tool(gateway, key, first_id, first_names["add"], {"a": 1, "b": 2})
        assert add.status_code == 200 and add.json()["isError"] is False, add.text
        assert add.json()["content"][0]["text"] == "3"
        multiply: Final = call_tool(gateway, key, first_id, first_names["multiply"], {"a": 2, "b": 3})
        assert multiply.status_code == 200 and multiply.json()["isError"] is False, multiply.text
        assert multiply.json()["content"][0]["text"] == "6"
        fail_name: Final = f"{shared_alias}-fail"
        denied: Final = call_tool(gateway, key, first_id, fail_name, {})
        assert denied.json().get("isError") is not False or denied.status_code != 200, denied.text
        assert len(tuple(item for item in first.drain() if item["body"].get("method") == "tools/call")) == 2
