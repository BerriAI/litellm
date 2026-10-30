import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value
from integration._support.mcp import McpCaller, mcp_peer, official_client_outcomes, register_mcp, tool_calls
from integration._support.process import owned_proxy


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
