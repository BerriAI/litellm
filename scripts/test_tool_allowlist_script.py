#!/usr/bin/env python3
"""
Standalone script to test tool allowlist enforcement and tool name extraction.

Run from repo root:
  poetry run python scripts/test_tool_allowlist_script.py

Or run the unit tests:
  poetry run pytest tests/test_litellm/proxy/test_tools_allowlist_enforcement.py -v
"""

import asyncio
import sys
from pathlib import Path

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_extraction():
    """Test extract_request_tool_names for each API shape."""
    from litellm.proxy.guardrails.tool_name_extraction import extract_request_tool_names

    cases = [
        ("OpenAI chat tools", "/v1/chat/completions", {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}),
        ("OpenAI chat functions", "/v1/chat/completions", {"functions": [{"name": "run_sql"}]}),
        ("OpenAI responses function", "/v1/responses", {"tools": [{"type": "function", "name": "get_current_weather"}]}),
        ("OpenAI responses MCP", "/v1/responses", {"tools": [{"type": "mcp", "server_label": "dmcp"}]}),
        ("Anthropic", "/v1/messages", {"tools": [{"name": "get_weather"}, {"name": "run_sql"}]}),
        ("Google generateContent", "/generate_content", {"tools": [{"functionDeclarations": [{"name": "schedule_meeting"}]}]}),
        ("MCP call_tool", "/mcp/call_tool", {"name": "my_tool", "arguments": {}}),
        ("Non-tool route", "/v1/embeddings", {"tools": [{"type": "function", "function": {"name": "x"}}]}),
    ]
    print("=== extract_request_tool_names(route, data) ===\n")
    for label, route, data in cases:
        names = extract_request_tool_names(route, data)
        print(f"  {label}: {names}")
    print()


async def test_check_tools_allowlist():
    """Test check_tools_allowlist with mock tokens."""
    from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import check_tools_allowlist

    def token(metadata=None, team_metadata=None):
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id="user",
            team_id="team",
            org_id=None,
            models=["*"],
            metadata=metadata or {},
            team_metadata=team_metadata or {},
        )

    print("=== check_tools_allowlist (auth) ===\n")

    # No allowlist -> pass
    await check_tools_allowlist(
        request_body={"tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        valid_token=token(),
        team_object=None,
        route="/v1/chat/completions",
    )
    print("  No allowlist, body has tools: PASS")

    # Allowed tool -> pass
    await check_tools_allowlist(
        request_body={"tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        valid_token=token(metadata={"allowed_tools": ["get_weather"]}),
        team_object=None,
        route="/v1/chat/completions",
    )
    print("  allowed_tools=['get_weather'], body has get_weather: PASS")

    # Disallowed tool -> raise
    try:
        await check_tools_allowlist(
            request_body={"tools": [{"type": "function", "function": {"name": "get_weather"}}]},
            valid_token=token(metadata={"allowed_tools": ["other_tool"]}),
            team_object=None,
            route="/v1/chat/completions",
        )
        print("  DISALLOWED: expected ProxyException")
    except ProxyException as e:
        if e.type == ProxyErrorTypes.tool_access_denied:
            print("  allowed_tools=['other_tool'], body has get_weather: PASS (raised tool_access_denied)")
        else:
            print(f"  Unexpected ProxyException type: {e.type}")
    except Exception as e:
        print(f"  Unexpected: {e}")

    # Team allowlist when key empty
    await check_tools_allowlist(
        request_body={"tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        valid_token=token(team_metadata={"allowed_tools": ["get_weather"]}),
        team_object=None,
        route="/v1/chat/completions",
    )
    print("  team_metadata.allowed_tools=['get_weather']: PASS")
    print()


def main():
    print("Tool allowlist / tool name extraction – script checks\n")
    test_extraction()
    asyncio.run(test_check_tools_allowlist())
    print("Done. For full unit tests run:")
    print("  poetry run pytest tests/test_litellm/proxy/test_tools_allowlist_enforcement.py -v")


if __name__ == "__main__":
    main()
