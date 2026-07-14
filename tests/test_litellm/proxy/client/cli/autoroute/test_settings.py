from litellm.proxy.client.cli.commands.autoroute.settings import merge_claude_settings_static_token


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
