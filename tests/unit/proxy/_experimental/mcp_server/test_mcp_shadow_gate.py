# bare-name mapping shadow gate (LD fork R2-23): a bare-name tool dispatch
# must not be hijackable by a foreign server advertising another server's
# prefix-qualified tool names.

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _manager_with_github_and_evil():
    manager = MCPServerManager()
    github = MCPServer(
        server_id="srv-github",
        name="github",
        alias="github",
        url="https://github-upstream.example/mcp",
        transport=MCPTransport.http,
    )
    evil = MCPServer(
        server_id="srv-evil",
        name="evil",
        alias="evil",
        url="https://evil-upstream.example/mcp",
        transport=MCPTransport.http,
    )
    manager.registry[github.server_id] = github
    manager.registry[evil.server_id] = evil
    return manager, github


def test_foreign_server_cannot_shadow_another_server_tool_name():
    """Server 'evil' lists a tool literally named 'github-delete_repo'; its
    registration overwrites github's prefix-qualified routing row
    ('github-delete_repo' -> 'evil'). The bare-name branch must refuse a
    candidate when the requested name carries a *different* registered
    server's known prefix, so the poisoned row cannot dispatch to evil."""
    manager, github = _manager_with_github_and_evil()
    routing = manager.tool_name_to_mcp_server_name_mapping
    routing["github-list_prs"] = "github"
    routing["github-delete_repo"] = "evil"  # evil overwrote github's row
    routing["delete_repo"] = "evil"
    routing["list_prs"] = "github"
    try:
        # the poisoned prefix-qualified row must not dispatch to evil
        assert manager.get_mcp_server_from_tool_name("github-delete_repo") is None
        assert manager.get_mcp_server_from_tool_name("github-list_prs") is github
        # a legit bare-name row keeps dispatching to its owner
        assert manager.get_mcp_server_from_tool_name("list_prs") is github
    finally:
        for key in ("github-list_prs", "github-delete_repo", "delete_repo", "list_prs"):
            routing.pop(key, None)
