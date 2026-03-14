"""Unit tests for litellm.setup_wizard — pure functions only, no network calls."""

from litellm.setup_wizard import SetupWizard, _yaml_escape

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
    "env_key": "AZURE_API_KEY",
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
        4000,
        "sk-master",
    )
    assert "model_list:" in config
    assert "model_name: gpt-4o" in config
    assert "model: gpt-4o" in config
    assert "api_key: os.environ/OPENAI_API_KEY" in config
    assert "master_key: sk-master" in config


def test_build_config_skipped_provider_omitted():
    """Provider with no key in env_vars should not appear in model_list."""
    config = SetupWizard._build_config(
        [_OPENAI, _ANTHROPIC],
        {"ANTHROPIC_API_KEY": "sk-ant-test"},  # OpenAI key missing
        4000,
        "sk-master",
    )
    assert "gpt-4o" not in config
    assert "claude-opus-4-6" in config


def test_build_config_env_vars_written_escaped():
    """API keys with special chars should be YAML-escaped."""
    config = SetupWizard._build_config(
        [_OPENAI],
        {"OPENAI_API_KEY": 'sk-ab"cd'},
        4000,
        "sk-master",
    )
    assert 'OPENAI_API_KEY: "sk-ab\\"cd"' in config


def test_build_config_does_not_mutate_env_vars():
    """_build_config must not modify the caller's env_vars dict."""
    env_vars = {
        "AZURE_API_KEY": "az-key",
        "_LITELLM_AZURE_API_BASE_AZURE": "https://my.azure.com",
        "_LITELLM_AZURE_DEPLOYMENT_AZURE": "my-deployment",
    }
    original_keys = set(env_vars.keys())
    SetupWizard._build_config([_AZURE], env_vars, 4000, "sk-master")
    assert set(env_vars.keys()) == original_keys


def test_build_config_azure_uses_deployment_name():
    env_vars = {
        "AZURE_API_KEY": "az-key",
        "_LITELLM_AZURE_API_BASE_AZURE": "https://my.azure.com",
        "_LITELLM_AZURE_DEPLOYMENT_AZURE": "my-gpt4o",
    }
    config = SetupWizard._build_config([_AZURE], env_vars, 4000, "sk-master")
    assert "model: azure/my-gpt4o" in config
    assert "model_name: azure-my-gpt4o" in config


def test_build_config_no_display_name_collision_openai_and_azure():
    """OpenAI gpt-4o and azure gpt-4o should get distinct model_name values."""
    env_vars = {
        "OPENAI_API_KEY": "sk-openai",
        "AZURE_API_KEY": "az-key",
        "_LITELLM_AZURE_DEPLOYMENT_AZURE": "gpt-4o",
    }
    config = SetupWizard._build_config([_OPENAI, _AZURE], env_vars, 4000, "sk-master")
    assert "model_name: gpt-4o" in config        # OpenAI
    assert "model_name: azure-gpt-4o" in config  # Azure — qualified


def test_build_config_ollama_no_api_key_line():
    """Ollama has no env_key — config should not contain an api_key line for it."""
    config = SetupWizard._build_config([_OLLAMA], {}, 4000, "sk-master")
    assert "ollama/llama3.2" in config
    assert "api_key:" not in config


def test_build_config_port_in_general_settings():
    """Port is not currently written to general_settings, but master_key is."""
    config = SetupWizard._build_config([_OPENAI], {"OPENAI_API_KEY": "k"}, 8080, "sk-m")
    assert "master_key: sk-m" in config


def test_build_config_internal_sentinel_keys_excluded():
    """_LITELLM_ prefixed sentinel keys must not appear in environment_variables."""
    env_vars = {
        "OPENAI_API_KEY": "sk-real",
        "_LITELLM_AZURE_API_BASE_AZURE": "https://x.azure.com",
    }
    config = SetupWizard._build_config([_OPENAI], env_vars, 4000, "sk-master")
    assert "_LITELLM_" not in config
