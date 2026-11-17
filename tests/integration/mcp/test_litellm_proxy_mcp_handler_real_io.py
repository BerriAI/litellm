import subprocess
import sys
import textwrap
from types import MappingProxyType
from typing import Final
from mcp.types import Tool as MCPTool
from litellm.types.mcp_server.mcp_toolset import MCPToolset
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def test_completion_with_function_tools_works_without_fastapi_installed():
    script = textwrap.dedent(
        """
        import sys

        class _FastapiBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fastapi" or fullname.startswith("fastapi."):
                    raise ModuleNotFoundError("No module named 'fastapi'")
                return None

        sys.meta_path.insert(0, _FastapiBlocker())

        import litellm

        response = litellm.completion(
            model="openai/gpt-5.5",
            messages=[{"role": "user", "content": "What is the weather in SF?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            mock_response="sunny",
        )
        assert response.choices[0].message.content == "sunny"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


TOOLSET_ROWS: Final = (
    MCPToolset(
        toolset_id="ts-granted", toolset_name="team-toolset", tools=[{"server_id": "srv-1", "tool_name": "add"}]
    ),
    MCPToolset(
        toolset_id="ts-other", toolset_name="other-toolset", tools=[{"server_id": "srv-2", "tool_name": "lookup"}]
    ),
    MCPToolset(toolset_id="ts-shouty", toolset_name="CALC", tools=[{"server_id": "srv-2", "tool_name": "lookup"}]),
    MCPToolset(toolset_id="ts-group", toolset_name="reference", tools=[{"server_id": "srv-1", "tool_name": "add"}]),
    MCPToolset(
        toolset_id="ts-prefixed",
        toolset_name="prefixed-toolset",
        tools=[{"server_id": "srv-1", "tool_name": "add"}, {"server_id": "srv-1", "tool_name": "calc-subtract"}],
    ),
    MCPToolset(
        toolset_id="ts-subtract", toolset_name="subtract-toolset", tools=[{"server_id": "srv-1", "tool_name": "subtract"}]
    ),
)
GATEWAY_SERVERS: Final = (
    MCPServer(server_id="srv-1", name="calc", alias="calc", transport=MCPTransport.http, access_groups=["wiki"]),
    MCPServer(server_id="srv-2", name="docs", alias="docs", transport=MCPTransport.http, access_groups=["reference"]),
)
UPSTREAM_TOOLS: Final = MappingProxyType(
    {
        "calc": (
            MCPTool(name="add", inputSchema={"type": "object"}),
            MCPTool(name="subtract", inputSchema={"type": "object"}),
        ),
        "docs": (MCPTool(name="lookup", inputSchema={"type": "object"}),),
    }
)
