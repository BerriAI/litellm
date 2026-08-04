"""validate_environment for friendliai — FRIENDLIAI_API_KEY or FRIENDLI_TOKEN."""

import litellm


def test_validate_environment_friendliai_missing(monkeypatch):
    monkeypatch.delenv("FRIENDLIAI_API_KEY", raising=False)
    monkeypatch.delenv("FRIENDLI_TOKEN", raising=False)
    result = litellm.validate_environment(model="friendliai/llama-3.1-8b-instruct")
    assert result["keys_in_environment"] is False
    assert "FRIENDLIAI_API_KEY" in result["missing_keys"]


def test_validate_environment_friendliai_with_api_key(monkeypatch):
    monkeypatch.delenv("FRIENDLI_TOKEN", raising=False)
    monkeypatch.setenv("FRIENDLIAI_API_KEY", "test-key")
    result = litellm.validate_environment(model="friendliai/llama-3.1-8b-instruct")
    assert result["keys_in_environment"] is True
    assert "FRIENDLIAI_API_KEY" not in result.get("missing_keys", [])


def test_validate_environment_friendliai_with_legacy_token(monkeypatch):
    monkeypatch.delenv("FRIENDLIAI_API_KEY", raising=False)
    monkeypatch.setenv("FRIENDLI_TOKEN", "legacy-token")
    result = litellm.validate_environment(model="friendliai/llama-3.1-8b-instruct")
    assert result["keys_in_environment"] is True
    assert result.get("missing_keys") == [] or "FRIENDLIAI_API_KEY" not in result["missing_keys"]
