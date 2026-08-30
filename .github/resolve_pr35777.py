from pathlib import Path

server_path = Path("litellm/proxy/_experimental/mcp_server/server.py")
text = server_path.read_text()

helper_anchor = "\n\ndef _jsonrpc_text_has_top_level_method(text: str) -> bool:\n"
helper = '''


def _request_tags_from_raw_headers(
    raw_headers: Mapping[str, str] | None,
) -> Sequence[str] | None:
    """Parse the caller's x-litellm-tags header with the shared proxy tag parser."""
    if not raw_headers:
        return None
    for key, value in raw_headers.items():
        if isinstance(key, str) and key.lower() == "x-litellm-tags" and value:
            return LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
                llm_router=None,
                headers={"x-litellm-tags": value},
                data={},
            )
    return None
'''
if "def _request_tags_from_raw_headers(" not in text:
    if helper_anchor not in text:
        raise SystemExit("helper anchor not found")
    text = text.replace(helper_anchor, helper + helper_anchor, 1)

old = "            effective_litellm_trace_id: Final = litellm_trace_id or get_chain_id_from_headers(raw_headers)\n            spend_logs_metadata: Final[dict[str, object]] = {\n"
new = "            effective_litellm_trace_id: Final = litellm_trace_id or get_chain_id_from_headers(raw_headers)\n            effective_request_tags: Final = request_tags or _request_tags_from_raw_headers(raw_headers)\n            spend_logs_metadata: Final[dict[str, object]] = {\n"
if "effective_request_tags: Final" not in text:
    if old not in text:
        raise SystemExit("list-tools logging anchor not found")
    text = text.replace(old, new, 1)

old_tags = '                    **({"tags": request_tags} if request_tags else {}),\n'
new_tags = '                    **({"tags": effective_request_tags} if effective_request_tags else {}),\n'
if old_tags in text:
    text = text.replace(old_tags, new_tags, 1)
elif new_tags not in text:
    raise SystemExit("tags metadata anchor not found")

server_path.write_text(text)

test_path = Path("tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server.py")
tests = test_path.read_text()
marker = "test_request_tags_from_raw_headers_reads_x_litellm_tags"
if marker not in tests:
    tests += r'''


@pytest.mark.parametrize(
    "raw_headers, expected",
    [
        (None, None),
        ({"mcp-session-id": "abc"}, None),
        ({"x-litellm-tags": ""}, None),
        (
            {"X-LiteLLM-Tags": "application:orders, service:checkout"},
            ["application:orders", "service:checkout"],
        ),
    ],
)
def test_request_tags_from_raw_headers_reads_x_litellm_tags(raw_headers, expected):
    from litellm.proxy._experimental.mcp_server.server import (
        _request_tags_from_raw_headers,
    )

    assert _request_tags_from_raw_headers(raw_headers) == expected


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_uses_x_litellm_tags_for_spend_logging():
    from litellm.proxy._experimental.mcp_server.server import (
        _get_tools_from_mcp_servers,
    )
    from mcp.types import Tool as MCPTool

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None
    server_a.tool_name_to_display_name = None
    server_a.tool_name_to_description = None

    tool_1 = MCPTool(
        name="server_a-tool_1",
        description="test tool",
        inputSchema={"type": "object"},
    )

    dummy_logging_obj = MagicMock()
    dummy_logging_obj.model_call_details = {"metadata": {"spend_logs_metadata": {}}}
    dummy_logging_obj.async_success_handler = AsyncMock()
    function_setup_kwargs = {}

    def _capture_function_setup(*_args, **kwargs):
        function_setup_kwargs.update(kwargs)
        return dummy_logging_obj, None

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            new=AsyncMock(return_value=[server_a]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=(None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_allowed_tools",
            side_effect=lambda tools, _server: tools,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_key_team_permissions",
            new=AsyncMock(side_effect=lambda tools, **_: tools),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.function_setup",
            side_effect=_capture_function_setup,
        ),
    ):
        mock_manager._get_tools_from_server = AsyncMock(return_value=[tool_1])

        listing = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_auth,
            mcp_auth_header=None,
            mcp_servers=["server_a"],
            mcp_server_auth_headers=None,
            raw_headers={"X-LiteLLM-Tags": "application:orders, service:checkout"},
            log_list_tools_to_spendlogs=True,
            list_tools_log_source="mcp_protocol",
        )

    assert listing.tools == [tool_1]
    assert function_setup_kwargs["metadata"]["tags"] == [
        "application:orders",
        "service:checkout",
    ]
'''
    test_path.write_text(tests)
