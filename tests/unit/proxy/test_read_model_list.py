"""Tests for litellm.proxy.read_model_list (Rust AI gateway config bridge)."""

from litellm.proxy.read_model_list import read_model_list


def test_read_model_list_resolves_os_environ(monkeypatch, tmp_path):
    """`os.environ/` markers in the model_list are resolved via ProxyConfig."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-resolved-123")
    config = tmp_path / "config.yaml"
    config.write_text(
        "model_list:\n"
        "  - model_name: gpt-realtime\n"
        "    litellm_params:\n"
        "      model: openai/gpt-realtime\n"
        "      api_key: os.environ/OPENAI_API_KEY\n"
    )

    model_list = read_model_list(str(config))

    assert len(model_list) == 1
    params = model_list[0]["litellm_params"]
    assert model_list[0]["model_name"] == "gpt-realtime"
    assert params["model"] == "openai/gpt-realtime"
    assert params["api_key"] == "sk-resolved-123"


def test_read_model_list_missing_key_returns_empty(tmp_path):
    """A config without a model_list yields an empty list, not an error."""
    config = tmp_path / "config.yaml"
    config.write_text("general_settings: {}\n")

    assert read_model_list(str(config)) == []
