from litellm.proxy.client.cli.commands.autoroute.settings import (
    ANTHROPIC_DEFAULT_MODEL_ENV_KEYS,
    merge_claude_settings_static_token,
)


def test_preserves_unrelated_top_level_keys():
    merged = merge_claude_settings_static_token({"theme": "dark"}, "http://127.0.0.1:4000", "token-abc")
    assert merged["theme"] == "dark"


def test_sets_base_url_and_auth_token():
    merged = merge_claude_settings_static_token({}, "http://127.0.0.1:4000", "token-abc")
    assert merged["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert merged["env"]["ANTHROPIC_AUTH_TOKEN"] == "token-abc"


def test_strips_trailing_slash_from_base_url():
    merged = merge_claude_settings_static_token({}, "http://127.0.0.1:4000/", "token-abc")
    assert merged["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"


def test_clears_existing_api_key_env_var():
    settings = {"env": {"ANTHROPIC_API_KEY": "sk-old"}}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert "ANTHROPIC_API_KEY" not in merged["env"]


def test_clears_existing_api_key_helper():
    settings = {"apiKeyHelper": "some-script.sh"}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert "apiKeyHelper" not in merged


def test_preserves_other_env_vars():
    settings = {"env": {"SOME_OTHER_VAR": "value"}}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert merged["env"]["SOME_OTHER_VAR"] == "value"


def test_sets_every_default_model_env_key_to_autorouter():
    merged = merge_claude_settings_static_token({}, "http://127.0.0.1:4000", "token-abc")
    for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS:
        assert merged["env"][key] == "autorouter"


def test_overrides_a_preexisting_default_model_env_var():
    settings = {"env": {"ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-4-8"}}
    merged = merge_claude_settings_static_token(settings, "http://127.0.0.1:4000", "token-abc")
    assert merged["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "autorouter"
