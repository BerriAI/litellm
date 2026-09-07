import json

from litellm.proxy.client.cli.commands.autoroute.settings import (
    ANTHROPIC_DEFAULT_MODEL_ENV_KEYS,
    SHUNT_BASH_ALLOW_RULES,
    merge_claude_settings_shunt_permissions,
    merge_claude_settings_static_token,
)


def test_preserves_unrelated_top_level_keys():
    merged = merge_claude_settings_static_token({"theme": "dark"}, "http://127.0.0.1:4000", "token-abc")
    assert merged["theme"] == "dark"


def test_preserves_unrelated_env_keys():
    settings = {"env": {"SOME_OTHER_VAR": "value"}}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert merged["env"]["SOME_OTHER_VAR"] == "value"


def test_sets_base_url_and_auth_token():
    merged = merge_claude_settings_static_token({}, "http://127.0.0.1:4000/", "token-abc")
    assert merged["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert merged["env"]["ANTHROPIC_AUTH_TOKEN"] == "token-abc"
    assert merged["env"]["ENABLE_TOOL_SEARCH"] == "true"


def test_preserves_existing_tool_search():
    settings = {"env": {"ENABLE_TOOL_SEARCH": "false"}}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert merged["env"]["ENABLE_TOOL_SEARCH"] == "false"


def test_drops_stray_api_key():
    settings = {"env": {"ANTHROPIC_API_KEY": "leaked-key"}}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert "ANTHROPIC_API_KEY" not in merged["env"]


def test_removes_existing_api_key_helper():
    settings = {"apiKeyHelper": "/usr/local/bin/lite auth print-token"}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert "apiKeyHelper" not in merged


def test_does_not_mutate_input():
    settings = {"env": {"FOO": "bar"}, "apiKeyHelper": "old-helper"}
    merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert settings == {"env": {"FOO": "bar"}, "apiKeyHelper": "old-helper"}


def test_forces_all_claude_code_default_model_tiers_to_the_autorouter():
    # A bare "*" model_name deployment looks like the obvious way to catch every request
    # regardless of which model Claude Code thinks it's using, but Router's auto-router
    # registry is keyed by the literal requested model string with no wildcard resolution
    # (litellm/router.py:10711-10717) -- so the only reliable way to make every one of Claude
    # Code's own tiers hit the auto-router is to override the env vars it reads per tier.
    merged = merge_claude_settings_static_token({}, "http://127.0.0.1:4000", "token-abc")
    for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS:
        assert merged["env"][key] == "autorouter"


def test_overrides_a_preexisting_default_model_env_var():
    settings = {"env": {"ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-4-8"}}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert merged["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "autorouter"


def test_adds_shunt_allow_rules_to_empty_settings():
    merged = merge_claude_settings_shunt_permissions({})
    assert tuple(merged["permissions"]["allow"]) == SHUNT_BASH_ALLOW_RULES


def test_preserves_unrelated_top_level_keys_for_permissions_merge():
    merged = merge_claude_settings_shunt_permissions({"theme": "dark"})
    assert merged["theme"] == "dark"


def test_preserves_existing_allow_rules():
    settings = {"permissions": {"allow": ["Bash(npm run *)"]}}
    merged = merge_claude_settings_shunt_permissions(settings)
    assert "Bash(npm run *)" in merged["permissions"]["allow"]
    for rule in SHUNT_BASH_ALLOW_RULES:
        assert rule in merged["permissions"]["allow"]


def test_preserves_deny_and_ask_rules():
    # Regression: a naive {**settings, "permissions": {...}} replaces the whole permissions
    # object, silently dropping deny/ask rules the caller already had.
    settings = {"permissions": {"deny": ["Bash(git push *)"], "ask": ["Bash(rm *)"]}}
    merged = merge_claude_settings_shunt_permissions(settings)
    assert merged["permissions"]["deny"] == ["Bash(git push *)"]
    assert merged["permissions"]["ask"] == ["Bash(rm *)"]
    for rule in SHUNT_BASH_ALLOW_RULES:
        assert rule in merged["permissions"]["allow"]


def test_does_not_duplicate_shunt_rules_already_present():
    settings = {"permissions": {"allow": list(SHUNT_BASH_ALLOW_RULES)}}
    merged = merge_claude_settings_shunt_permissions(settings)
    assert tuple(merged["permissions"]["allow"]) == SHUNT_BASH_ALLOW_RULES


def test_merged_settings_survive_the_json_round_trip_commands_py_writes():
    # Regression: the merge returns MappingProxyType nested at every level, which the JSON
    # encoder rejects without `default=dict` -- the exact call commands.py makes. Without it
    # `lite autoroute up` raises instead of writing settings.json.
    settings = {"permissions": {"allow": ["Bash(npm run *)"], "deny": ["Bash(rm *)"]}, "theme": "dark"}
    merged = merge_claude_settings_shunt_permissions(settings)
    reloaded = json.loads(json.dumps(merged, default=dict))
    assert reloaded["theme"] == "dark"
    assert reloaded["permissions"]["deny"] == ["Bash(rm *)"]
    assert reloaded["permissions"]["allow"] == ["Bash(npm run *)", *SHUNT_BASH_ALLOW_RULES]


def test_does_not_mutate_input_for_permissions_merge():
    settings = {"permissions": {"allow": ["Bash(npm run *)"], "deny": ["Bash(rm *)"]}}
    merge_claude_settings_shunt_permissions(settings)
    assert settings == {"permissions": {"allow": ["Bash(npm run *)"], "deny": ["Bash(rm *)"]}}
