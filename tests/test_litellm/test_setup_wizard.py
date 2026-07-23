"""Unit tests for litellm.setup_wizard — pure functions only, no network calls."""

from litellm.exceptions import AuthenticationError, RateLimitError
from litellm.setup_wizard import (
    SetupWizard,
    _KeyInvalid,
    _KeyUnverified,
    _KeyValid,
    _yaml_escape,
)

# ---------------------------------------------------------------------------
# _yaml_escape
# ---------------------------------------------------------------------------


def test_yaml_escape_plain():
    assert _yaml_escape("sk-abc123") == "sk-abc123"


def test_yaml_escape_double_quote():
    assert _yaml_escape('sk-ab"cd') == 'sk-ab\\"cd'


def test_yaml_escape_backslash():
    assert _yaml_escape("sk-ab\\cd") == "sk-ab\\\\cd"


def test_yaml_escape_combined():
    assert _yaml_escape('ab\\"cd') == 'ab\\\\\\"cd'


def test_yaml_escape_newline():
    assert _yaml_escape("sk-abc\ndef") == "sk-abc\\ndef"


def test_yaml_escape_carriage_return():
    assert _yaml_escape("sk-abc\rdef") == "sk-abc\\rdef"


def test_yaml_escape_tab():
    assert _yaml_escape("sk-abc\tdef") == "sk-abc\\tdef"


# ---------------------------------------------------------------------------
# SetupWizard._build_config
# ---------------------------------------------------------------------------

_OPENAI = {
    "id": "openai",
    "name": "OpenAI",
    "env_key": "OPENAI_API_KEY",
    "models": ["gpt-4o", "gpt-4o-mini"],
    "test_model": "gpt-4o-mini",
}

_ANTHROPIC = {
    "id": "anthropic",
    "name": "Anthropic",
    "env_key": "ANTHROPIC_API_KEY",
    "models": ["claude-opus-4-6"],
    "test_model": "claude-haiku-4-5-20251001",
}

_AZURE = {
    "id": "azure",
    "name": "Azure OpenAI",
    "env_key": "AZURE_AI_API_KEY",
    "models": [],
    "test_model": None,
    "needs_api_base": True,
    "api_base_hint": "https://<resource>.openai.azure.com/",
    "api_version": "2024-07-01-preview",
}

_OLLAMA = {
    "id": "ollama",
    "name": "Ollama",
    "env_key": None,
    "models": ["ollama/llama3.2"],
    "test_model": None,
    "api_base": "http://localhost:11434",
}


def test_build_config_basic_openai():
    config = SetupWizard._build_config(
        [_OPENAI],
        {"OPENAI_API_KEY": "sk-test"},
        "sk-master",
    )
    assert "model_list:" in config
    assert "model_name: gpt-4o" in config
    assert "model: gpt-4o" in config
    assert "api_key: os.environ/OPENAI_API_KEY" in config
    assert 'master_key: "sk-master"' in config


def test_build_config_skipped_provider_omitted():
    """Provider with no key in env_vars should not appear in model_list."""
    config = SetupWizard._build_config(
        [_OPENAI, _ANTHROPIC],
        {"ANTHROPIC_API_KEY": "sk-ant-test"},  # OpenAI key missing
        "sk-master",
    )
    assert "gpt-4o" not in config
    assert "claude-opus-4-6" in config


def test_build_config_env_vars_written_escaped():
    """API keys with special chars should be YAML-escaped."""
    config = SetupWizard._build_config(
        [_OPENAI],
        {"OPENAI_API_KEY": 'sk-ab"cd'},
        "sk-master",
    )
    assert 'OPENAI_API_KEY: "sk-ab\\"cd"' in config


def test_build_config_master_key_quoted():
    """master_key must be quoted in YAML to handle special characters."""
    config = SetupWizard._build_config(
        [_OPENAI],
        {"OPENAI_API_KEY": "sk-test"},
        'sk-master"special',
    )
    assert 'master_key: "sk-master\\"special"' in config


def test_build_config_does_not_mutate_env_vars():
    """_build_config must not modify the caller's env_vars dict."""
    env_vars = {
        "AZURE_AI_API_KEY": "az-key",
        "_LITELLM_AZURE_AI_API_BASE_AZURE": "https://my.azure.com",
        "_LITELLM_AZURE_DEPLOYMENT_AZURE": "my-deployment",
    }
    original_keys = set(env_vars.keys())
    SetupWizard._build_config([_AZURE], env_vars, "sk-master")
    assert set(env_vars.keys()) == original_keys


def test_build_config_azure_uses_deployment_name():
    env_vars = {
        "AZURE_AI_API_KEY": "az-key",
        "_LITELLM_AZURE_AI_API_BASE_AZURE": "https://my.azure.com",
        "_LITELLM_AZURE_DEPLOYMENT_AZURE": "my-gpt4o",
    }
    config = SetupWizard._build_config([_AZURE], env_vars, "sk-master")
    assert "model: azure/my-gpt4o" in config
    assert "model_name: azure-my-gpt4o" in config
    # api_base must be quoted to survive YAML special chars
    assert 'api_base: "https://my.azure.com"' in config


def test_build_config_azure_no_deployment_skipped():
    """Azure without a deployment name should emit nothing (not fallback to gpt-4o)."""
    env_vars = {"AZURE_AI_API_KEY": "az-key"}  # no deployment sentinel
    config = SetupWizard._build_config([_AZURE], env_vars, "sk-master")
    # No azure model entry should be emitted when deployment name is absent
    assert "model: azure/" not in config


def test_build_config_no_display_name_collision_openai_and_azure():
    """OpenAI gpt-4o and azure gpt-4o should get distinct model_name values."""
    env_vars = {
        "OPENAI_API_KEY": "sk-openai",
        "AZURE_AI_API_KEY": "az-key",
        "_LITELLM_AZURE_DEPLOYMENT_AZURE": "gpt-4o",
    }
    config = SetupWizard._build_config([_OPENAI, _AZURE], env_vars, "sk-master")
    assert "model_name: gpt-4o" in config  # OpenAI
    assert "model_name: azure-gpt-4o" in config  # Azure — qualified


def test_build_config_ollama_no_api_key_line():
    """Ollama has no env_key — config should not contain an api_key line for it."""
    config = SetupWizard._build_config([_OLLAMA], {}, "sk-master")
    assert "ollama/llama3.2" in config
    assert "api_key:" not in config


def test_build_config_master_key_in_general_settings():
    """master_key is written to general_settings."""
    config = SetupWizard._build_config([_OPENAI], {"OPENAI_API_KEY": "k"}, "sk-m")
    assert 'master_key: "sk-m"' in config


def test_build_config_internal_sentinel_keys_excluded():
    """_LITELLM_ prefixed sentinel keys must not appear in environment_variables."""
    env_vars = {
        "OPENAI_API_KEY": "sk-real",
        "_LITELLM_AZURE_AI_API_BASE_AZURE": "https://x.azure.com",
    }
    config = SetupWizard._build_config([_OPENAI], env_vars, "sk-master")
    assert "_LITELLM_" not in config


# ---------------------------------------------------------------------------
# SetupWizard._classify_key
# ---------------------------------------------------------------------------


def test_classify_key_valid_on_success():
    def fake_completion(**kwargs):
        assert kwargs["model"] == "gemini/gemini-3.5-flash"
        assert kwargs["api_key"] == "good-key"
        return object()

    result = SetupWizard._classify_key("gemini/gemini-3.5-flash", "good-key", fake_completion)
    assert isinstance(result, _KeyValid)


def test_classify_key_invalid_only_on_auth_error():
    def fake_completion(**kwargs):
        raise AuthenticationError(
            message="API key not valid",
            llm_provider="gemini",
            model="gemini/gemini-3.5-flash",
        )

    result = SetupWizard._classify_key("gemini/gemini-3.5-flash", "bad-key", fake_completion)
    assert isinstance(result, _KeyInvalid)


def test_classify_key_unverified_on_non_auth_error():
    """A valid key hitting a rate limit must NOT be reported as invalid; the
    real reason is surfaced so users can debug it."""

    def fake_completion(**kwargs):
        raise RateLimitError(
            message="Resource has been exhausted (e.g. check quota)",
            llm_provider="gemini",
            model="gemini/gemini-3.5-flash",
        )

    result = SetupWizard._classify_key("gemini/gemini-3.5-flash", "valid-but-throttled", fake_completion)
    assert isinstance(result, _KeyUnverified)
    assert "RateLimitError" in result.reason
    assert "check quota" in result.reason


def test_classify_key_unverified_on_generic_error():
    def fake_completion(**kwargs):
        raise ValueError("model gemini/gemini-3.5-flash not found for this key")

    result = SetupWizard._classify_key("gemini/gemini-3.5-flash", "valid-key", fake_completion)
    assert isinstance(result, _KeyUnverified)
    assert "ValueError" in result.reason
    assert "not found" in result.reason


def _auth_error():
    return AuthenticationError(
        message="invalid key",
        llm_provider="gemini",
        model="gemini/gemini-3.5-flash",
    )


_GEMINI = {
    "name": "Google Gemini",
    "env_key": "GEMINI_API_KEY",
    "key_hint": "AIza...",
    "test_model": "gemini/gemini-3.5-flash",
}


def test_validate_and_report_skips_when_no_test_model():
    """Providers without a test_model (Azure/Bedrock/Ollama) return the key untouched."""
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1

    key = SetupWizard._validate_and_report({"name": "Azure", "test_model": None}, "az-key", fake_completion)
    assert key == "az-key"
    assert calls["n"] == 0  # no validation attempted


def test_validate_and_report_valid_returns_key(capsys):
    key = SetupWizard._validate_and_report(_GEMINI, "good-key", lambda **_: object())
    assert key == "good-key"
    assert "connected successfully" in capsys.readouterr().out


def test_validate_and_report_unverified_surfaces_reason(capsys, monkeypatch):
    """A non-auth failure must print the real reason, not 'invalid API key'."""
    monkeypatch.setattr("builtins.input", lambda *_: "n")  # decline re-entry

    def fake_completion(**_):
        raise RateLimitError(
            message="Resource has been exhausted (check quota)",
            llm_provider="gemini",
            model="gemini/gemini-3.5-flash",
        )

    key = SetupWizard._validate_and_report(_GEMINI, "valid-but-throttled", fake_completion)
    out = capsys.readouterr().out
    assert key == "valid-but-throttled"
    assert "could not verify key" in out
    assert "check quota" in out
    assert "invalid API key" not in out


def test_validate_and_report_invalid_key_prompt(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "n")  # decline re-entry

    key = SetupWizard._validate_and_report(_GEMINI, "bad-key", lambda **_: (_ for _ in ()).throw(_auth_error()))
    out = capsys.readouterr().out
    assert key == "bad-key"
    assert "invalid API key" in out


def test_validate_and_report_reentry_accepts_new_key(monkeypatch):
    """On re-entry the newly typed key is validated and returned once it works."""
    inputs = iter(["y", "good-key"])  # yes re-enter, then the replacement key
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    seen = {"keys": []}

    def fake_completion(**kwargs):
        seen["keys"].append(kwargs["api_key"])
        if kwargs["api_key"] == "bad-key":
            raise _auth_error()
        return object()

    key = SetupWizard._validate_and_report(_GEMINI, "bad-key", fake_completion)
    assert key == "good-key"
    assert seen["keys"] == ["bad-key", "good-key"]


def test_run_setup_wizard_enables_debug_when_requested(monkeypatch):
    import litellm
    import litellm.setup_wizard as wiz

    calls = {"debug": 0, "run": 0}
    monkeypatch.setattr(litellm, "_turn_on_debug", lambda: calls.__setitem__("debug", calls["debug"] + 1))
    monkeypatch.setattr(wiz.SetupWizard, "run", staticmethod(lambda: calls.__setitem__("run", calls["run"] + 1)))

    wiz.run_setup_wizard(debug=True)
    assert calls == {"debug": 1, "run": 1}

    calls["debug"] = 0
    calls["run"] = 0
    wiz.run_setup_wizard(debug=False)
    assert calls == {"debug": 0, "run": 1}
