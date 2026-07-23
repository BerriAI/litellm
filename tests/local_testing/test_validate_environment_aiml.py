import litellm


def test_validate_environment_aiml_missing(monkeypatch):
    for k in ("AIML_API_KEY", "AIMLAPI_KEY"):
        monkeypatch.delenv(k, raising=False)
    result = litellm.validate_environment("aiml/gpt-4o")
    assert result["keys_in_environment"] is False
    assert "AIML_API_KEY" in result["missing_keys"]


def test_validate_environment_aiml_present(monkeypatch):
    monkeypatch.setenv("AIML_API_KEY", "sk-test")
    monkeypatch.delenv("AIMLAPI_KEY", raising=False)
    result = litellm.validate_environment("aiml/gpt-4o")
    assert result["keys_in_environment"] is True
    assert result["missing_keys"] == []


def test_validate_environment_aiml_alt_key(monkeypatch):
    monkeypatch.delenv("AIML_API_KEY", raising=False)
    monkeypatch.setenv("AIMLAPI_KEY", "sk-alt")
    result = litellm.validate_environment("aiml/gpt-4o")
    assert result["keys_in_environment"] is True
    assert result["missing_keys"] == []
