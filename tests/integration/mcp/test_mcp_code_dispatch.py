import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value
from integration._support.mcp import (
    McpCaller,
    echo_tool,
    mcp_peer,
    official_client_outcomes,
    register_mcp,
    scripted_peer,
    tool_calls,
)
from integration._support.process import owned_proxy
from pydantic import JsonValue, TypeAdapter

SEARCH_RESULTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


@pytest.fixture(scope="module")
def code_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("mcp-code-dispatch")
    configuration: Final = JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    path: Final = directory / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **configuration,
                "litellm_settings": {**object_value(configuration["litellm_settings"]), "mcp_code_mode": True},
            }
        )
    )
    with (
        gateway_from_environment() as existing,
        owned_proxy(existing, directory, {}, config=path, workers=2) as gateway,
    ):
        yield gateway


def test_code_calls_chain_with_caller_permissions(code_gateway: Gateway) -> None:
    with mcp_peer() as allowed, mcp_peer() as denied, code_gateway.scenario() as scenario:
        alias: Final = "code" + uuid.uuid4().hex[:8]
        denied_alias: Final = "denied" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, allowed, alias)
        register_mcp(scenario, denied, denied_alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]}, max_parallel_requests=1)
        caller: Final = McpCaller(code_gateway, key, "mcp")
        assert "litellm_execute_code" in caller.list_tools().tools
        allowed.drain()
        denied.drain()
        result: Final = caller.call(
            "litellm_execute_code",
            {
                "code": f'''
            const sum = await tools.call_tool({{tool_id: "{alias}-add", arguments: {{a: 17, b: 25}}}});
            text(await tools.call_tool({{tool_id: "{alias}-multiply", arguments: {{a: Number(sum.content[0].text), b: 3}}}}));
            text(await tools.call_tool({{tool_id: "{denied_alias}-add", arguments: {{a: 1, b: 2}}}}));
        '''
            },
        )
        assert result.ok and result.text is not None, result.raw
        body: Final = JSON_OBJECT.validate_json(result.text)
        output: Final = body["output"]
        assert isinstance(output, list) and len(output) == 2, body
        successful: Final = object_value(output[0])
        refused: Final = object_value(output[1])
        content: Final = successful["content"]
        assert isinstance(content, list) and object_value(content[0])["text"] == "126", successful
        assert successful.get("isError", False) is False
        assert refused["isError"] is True
        assert len(tool_calls(allowed.drain())) == 2
        assert tool_calls(denied.drain()) == ()


def test_code_timeout_returns_error_and_releases_caller_slot(code_gateway: Gateway) -> None:
    with mcp_peer() as peer, code_gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "timeout" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]}, max_parallel_requests=1)
        _, result = official_client_outcomes(
            code_gateway, key, "/mcp", "litellm_execute_code", {"code": "while (true) {}"}
        )
        assert result.status == 200 and not result.ok and result.text is not None, result
        body: Final = JSON_OBJECT.validate_json(result.text)
        error: Final = body["error"]
        assert isinstance(error, str) and "timed out" in error, body
        assert body["calls_started"] == 0
        recovery: Final = McpCaller(code_gateway, key, "mcp").call("litellm_execute_code", {"code": "text(42)"})
        assert recovery.ok and recovery.text is not None, recovery
        assert JSON_OBJECT.validate_json(recovery.text)["output"] == [42]


def test_rest_server_id_preserves_upstream_reserved_name(code_gateway: Gateway) -> None:
    with scripted_peer(echo_tool("litellm_execute_code")) as peer, code_gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "reserved" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        arguments: Final = {"code": "qa_upstream_marker"}
        peer.drain()
        upstream: Final = McpCaller(code_gateway, key, "rest").call("litellm_execute_code", arguments, identity)
        assert upstream.ok and upstream.text is not None, upstream
        assert JSON_OBJECT.validate_json(upstream.text) == arguments
        observed: Final = tool_calls(peer.drain())
        assert len(observed) == 1
        request: Final = JSON_OBJECT.validate_python(observed[0]["body"])
        assert object_value(request["params"])["arguments"] == arguments
        gateway_code: Final = McpCaller(code_gateway, key, "rest").call("litellm_execute_code", {"code": "text(42)"})
        assert gateway_code.ok and gateway_code.text is not None, gateway_code
        assert JSON_OBJECT.validate_json(gateway_code.text)["output"] == [42]
        assert tool_calls(peer.drain()) == ()


def test_invalid_rest_code_returns_tool_error_and_releases_slot(code_gateway: Gateway) -> None:
    with mcp_peer() as peer, code_gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "invalid" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]}, max_parallel_requests=1)
        caller: Final = McpCaller(code_gateway, key, "rest")
        invalid: Final = caller.call("litellm_execute_code", {"code": 42})
        assert invalid.status == 200 and not invalid.ok, invalid
        valid: Final = caller.call("litellm_execute_code", {"code": "text(42)"})
        assert valid.ok and valid.text is not None, valid
        assert JSON_OBJECT.validate_json(valid.text)["output"] == [42]
        assert tool_calls(peer.drain()) == ()


def _remaining_server_calls(gateway: Gateway, key: str, identity: str) -> int:
    responses: Final = tuple(
        gateway.request(
            "POST",
            "/mcp-rest/tools/call",
            {"server_id": identity, "name": "add", "arguments": {"a": 20, "b": 22}},
            key=key,
        )
        for _ in range(7)
    )
    assert all(response.status_code in (200, 429) for response in responses)
    return sum(response.status_code == 200 for response in responses)


def _server_budget_after_call(gateway: Gateway, *, code: bool) -> tuple[int, int]:
    with mcp_peer() as target, mcp_peer() as other, gateway.scenario() as scenario:
        alias: Final = "target" + uuid.uuid4().hex[:8]
        target_id: Final = register_mcp(scenario, target, alias, rpm=6)
        other_id: Final = register_mcp(scenario, other, "other" + uuid.uuid4().hex[:8], rpm=6)
        key: Final = scenario.key(object_permission={"mcp_servers": [target_id, other_id]})
        _, search = official_client_outcomes(gateway, key, "/mcp/proxy", "search_tools", {"query": "add", "top_k": 10})
        assert search.ok and search.text is not None, search
        matches: Final = SEARCH_RESULTS.validate_json(search.text)
        identity: Final = next(hit["tool_id"] for hit in matches if hit["name"] == f"{alias}-add")
        assert isinstance(identity, str)
        target.drain()
        other.drain()
        arguments: Final = (
            {"code": f"text(await tools.call_tool({{tool_id:{json.dumps(identity)},arguments:{{a:20,b:22}}}}))"}
            if code
            else {"tool_id": identity, "arguments": {"a": 20, "b": 22}}
        )
        _, result = official_client_outcomes(
            gateway, key, "/mcp/proxy", "litellm_execute_code" if code else "call_tool", arguments
        )
        assert result.ok and result.text is not None, result
        if code:
            output: Final = JSON_OBJECT.validate_json(result.text)["output"]
            assert isinstance(output, list) and len(output) == 1
            assert object_value(output[0]).get("isError", False) is False
        else:
            assert result.text == "42"
        assert len(tool_calls(target.drain())) == 1
        assert tool_calls(other.drain()) == ()
        return _remaining_server_calls(gateway, key, target_id), _remaining_server_calls(gateway, key, other_id)


def test_code_preserves_direct_server_rpm_for_selected_and_unrelated_servers(code_gateway: Gateway) -> None:
    direct: Final = _server_budget_after_call(code_gateway, code=False)
    nested: Final = _server_budget_after_call(code_gateway, code=True)
    assert all(remaining > 0 for remaining in direct)
    assert nested == direct
