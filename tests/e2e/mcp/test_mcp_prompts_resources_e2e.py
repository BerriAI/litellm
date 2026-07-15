"""Live e2e: the MCP gateway's prompt and resource primitives.

Covers mcp.list_prompts.api_key.succeeds, mcp.get_prompt.api_key.succeeds,
mcp.list_resources.api_key.succeeds, and mcp.read_resource.api_key.succeeds,
against the stub's /conformance mount (tests/e2e/mcp/stub/), the one upstream
that serves all three MCP primitives.

The gateway's namespacing contract differs per primitive, and the assertions
pin it exactly as observed live: prompt names are alias-prefixed like tool
names (and must be fetched by the prefixed name), while resource URIs pass
through unprefixed (only the resource's display name gains the prefix). The
prompt-rendering assertion feeds unique per-run argument values through
prompts/get and requires them back verbatim in the rendered text, so a cached
or canned response cannot pass; the resource assertion requires the exact
fixture body.

The binary-resource read (test://static-binary) is deliberately NOT asserted
here: the gateway currently drops the base64 blob field on read-through
(verified against the stub directly, which serves it), and that gap is pinned
in the conformance suite baseline (test_mcp_conformance_e2e.py) instead.
"""

from __future__ import annotations

import pytest

from e2e_config import MCP_STUB_CONFORMANCE_URL, unique_marker
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import KeyGenerateBody, McpServerCreateBody

pytestmark = pytest.mark.e2e


class TestMcpPromptsAndResources:
    """A registered server's prompts and resources are listed, fetched, and
    read through the gateway with the same lifecycle contract as tools."""

    @pytest.mark.covers("mcp.list_prompts.api_key.succeeds")
    @pytest.mark.covers("mcp.get_prompt.api_key.succeeds")
    def test_prompts_list_and_render_with_arguments(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        alias = f"e2emcpprompts{unique_marker()}"
        created = client.create_server(
            McpServerCreateBody(alias=alias, url=MCP_STUB_CONFORMANCE_URL, allow_all_keys=True)
        )
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.alias == alias
        assert stored.url == MCP_STUB_CONFORMANCE_URL

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))
        headers = {"x-litellm-api-key": f"Bearer {key}"}
        _ = client.poll_tool_names(alias, headers)

        listed = client.list_prompts(alias, headers)
        expected = (
            (f"{alias}-test_prompt_with_arguments", ("arg1", "arg2")),
            (f"{alias}-test_simple_prompt", ()),
        )
        assert listed == expected, f"gateway listed prompts {listed}, expected exactly {expected}"

        first_value = f"e2e-{unique_marker()}"
        second_value = f"e2e-{unique_marker()}"
        rendered = client.get_prompt(
            alias, headers, f"{alias}-test_prompt_with_arguments", {"arg1": first_value, "arg2": second_value}
        )
        assert rendered == f"Prompt rendered with arg1={first_value} and arg2={second_value}"

    @pytest.mark.covers("mcp.list_resources.api_key.succeeds")
    @pytest.mark.covers("mcp.read_resource.api_key.succeeds")
    def test_resources_list_and_read_text(self, client: McpClient, resources: ResourceManager) -> None:
        alias = f"e2emcpresources{unique_marker()}"
        created = client.create_server(
            McpServerCreateBody(alias=alias, url=MCP_STUB_CONFORMANCE_URL, allow_all_keys=True)
        )
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.alias == alias
        assert stored.url == MCP_STUB_CONFORMANCE_URL

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))
        headers = {"x-litellm-api-key": f"Bearer {key}"}
        _ = client.poll_tool_names(alias, headers)

        listed = client.list_resources(alias, headers)
        expected = (
            ("test://static-binary", f"{alias}-static_binary_resource"),
            ("test://static-text", f"{alias}-static_text_resource"),
        )
        assert listed == expected, f"gateway listed resources {listed}, expected exactly {expected}"

        body = client.read_resource(alias, headers, "test://static-text")
        assert body == "Static text resource from the e2e conformance stub"
