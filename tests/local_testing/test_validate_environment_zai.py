"""validate_environment must report missing ZAI_API_KEY for zai/* models."""

import os

import litellm


def test_validate_environment_zai_missing_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    result = litellm.validate_environment(model="zai/glm-5.1")
    assert result["keys_in_environment"] is False
    assert "ZAI_API_KEY" in result["missing_keys"]


def test_validate_environment_zai_with_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    result = litellm.validate_environment(model="zai/glm-5.1")
    assert result["keys_in_environment"] is True
    assert "ZAI_API_KEY" not in result.get("missing_keys", [])
