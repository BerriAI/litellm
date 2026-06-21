"""Unit tests for the sandbox-tool registry."""

from litellm.sandbox import sandbox_tools


def _reset():
    sandbox_tools.clear_sandbox_tools()


def test_register_resolves_provider_key_and_base():
    _reset()
    sandbox_tools.register_sandbox_tools(
        [
            {
                "sandbox_tool_name": "e2b_default",
                "litellm_params": {
                    "sandbox_provider": "e2b",
                    "api_key": "sk-literal",
                    "api_base": "https://sandbox.internal",
                },
            }
        ]
    )

    resolved = sandbox_tools.resolve_sandbox_tool("e2b_default")
    assert resolved == {
        "sandbox_provider": "e2b",
        "api_key": "sk-literal",
        "api_base": "https://sandbox.internal",
    }
    _reset()


def test_register_clears_stale_entries_on_reload():
    """A tool removed from the config must not survive a re-registration."""
    _reset()
    sandbox_tools.register_sandbox_tools(
        [
            {
                "sandbox_tool_name": "old",
                "litellm_params": {"sandbox_provider": "e2b"},
            }
        ]
    )
    assert sandbox_tools.resolve_sandbox_tool("old") is not None

    sandbox_tools.register_sandbox_tools(
        [
            {
                "sandbox_tool_name": "new",
                "litellm_params": {"sandbox_provider": "e2b"},
            }
        ]
    )

    assert sandbox_tools.resolve_sandbox_tool("new") is not None
    assert (
        sandbox_tools.resolve_sandbox_tool("old") is None
    ), "stale tool must be gone after the config is reloaded"
    _reset()


def test_register_resolves_secret_from_env(monkeypatch):
    _reset()
    monkeypatch.setenv("MY_SANDBOX_KEY", "sk-from-env")
    sandbox_tools.register_sandbox_tools(
        [
            {
                "sandbox_tool_name": "e2b_default",
                "litellm_params": {
                    "sandbox_provider": "e2b",
                    "api_key": "os.environ/MY_SANDBOX_KEY",
                },
            }
        ]
    )

    resolved = sandbox_tools.resolve_sandbox_tool("e2b_default")
    assert resolved is not None
    assert resolved["api_key"] == "sk-from-env"
    assert resolved["api_base"] is None
    _reset()


def test_resolve_unknown_returns_none():
    _reset()
    assert sandbox_tools.resolve_sandbox_tool("nope") is None
