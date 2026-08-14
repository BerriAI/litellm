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


def test_register_empty_list_clears_removed_tools():
    """Reloading a config with sandbox_tools removed (the proxy passes an empty
    list) must drop previously registered credentials from the process."""
    _reset()
    sandbox_tools.register_sandbox_tools(
        [
            {
                "sandbox_tool_name": "e2b_default",
                "litellm_params": {"sandbox_provider": "e2b", "api_key": "sk-x"},
            }
        ]
    )
    assert sandbox_tools.resolve_sandbox_tool("e2b_default") is not None

    sandbox_tools.register_sandbox_tools([])

    assert (
        sandbox_tools.resolve_sandbox_tool("e2b_default") is None
    ), "removing sandbox_tools from config must clear stale credentials"
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


def test_register_skips_malformed_entries_without_crashing():
    """A single malformed entry (missing sandbox_tool_name, or not a dict) must
    not crash registration during proxy startup/hot-reload; valid entries in the
    same list must still register."""
    _reset()
    sandbox_tools.register_sandbox_tools(
        [
            {"litellm_params": {"sandbox_provider": "e2b"}},  # missing name
            "not-a-dict",  # wrong type
            {"sandbox_tool_name": "", "litellm_params": {}},  # empty name
            {
                "sandbox_tool_name": "good",
                "litellm_params": {"sandbox_provider": "e2b"},
            },
        ]
    )

    assert sandbox_tools.resolve_sandbox_tool("good") is not None
    assert sandbox_tools.resolve_sandbox_tool("") is None
    assert set(sandbox_tools._SANDBOX_TOOL_REGISTRY) == {"good"}
    _reset()


def test_register_skips_entry_missing_sandbox_provider():
    """An entry with a name but no sandbox_provider must be skipped at
    registration so it cannot later resolve and call acreate_sandbox(provider=None),
    which fails with a cryptic runtime error instead of a clear startup warning."""
    _reset()
    sandbox_tools.register_sandbox_tools(
        [
            {"sandbox_tool_name": "no_provider", "litellm_params": {"api_key": "sk-x"}},
            {
                "sandbox_tool_name": "null_provider",
                "litellm_params": {"sandbox_provider": None},
            },
            {
                "sandbox_tool_name": "good",
                "litellm_params": {"sandbox_provider": "e2b"},
            },
        ]
    )

    assert sandbox_tools.resolve_sandbox_tool("no_provider") is None
    assert sandbox_tools.resolve_sandbox_tool("null_provider") is None
    assert set(sandbox_tools._SANDBOX_TOOL_REGISTRY) == {"good"}
    _reset()


def test_register_swaps_registry_atomically():
    """register_sandbox_tools must replace the registry in one rebind so a
    concurrent resolve never observes a half-populated or transiently empty
    registry between clearing and repopulating."""
    _reset()
    sandbox_tools.register_sandbox_tools(
        [{"sandbox_tool_name": "a", "litellm_params": {"sandbox_provider": "e2b"}}]
    )
    before = sandbox_tools._SANDBOX_TOOL_REGISTRY

    sandbox_tools.register_sandbox_tools(
        [
            {"sandbox_tool_name": "b", "litellm_params": {"sandbox_provider": "e2b"}},
            {"sandbox_tool_name": "c", "litellm_params": {"sandbox_provider": "e2b"}},
        ]
    )
    after = sandbox_tools._SANDBOX_TOOL_REGISTRY

    assert after is not before, "the registry must be replaced, not mutated in place"
    assert set(after) == {"b", "c"}
    assert "a" not in after
    _reset()
