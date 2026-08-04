import os

import litellm


def test_validate_environment_galadriel_missing_key(monkeypatch):
    monkeypatch.delenv("GALADRIEL_API_KEY", raising=False)
    result = litellm.validate_environment(model="galadriel/gpt-4o")
    assert result["keys_in_environment"] is False
    assert "GALADRIEL_API_KEY" in result["missing_keys"]


def test_validate_environment_galadriel_present(monkeypatch):
    monkeypatch.setenv("GALADRIEL_API_KEY", "sk-test")
    result = litellm.validate_environment(model="galadriel/gpt-4o")
    assert result["keys_in_environment"] is True
    assert result["missing_keys"] == []


def test_validate_environment_bytez_missing_key(monkeypatch):
    monkeypatch.delenv("BYTEZ_API_KEY", raising=False)
    result = litellm.validate_environment(model="bytez/google/gemma-2b")
    assert result["keys_in_environment"] is False
    assert "BYTEZ_API_KEY" in result["missing_keys"]


def test_validate_environment_bytez_present(monkeypatch):
    monkeypatch.setenv("BYTEZ_API_KEY", "sk-test")
    result = litellm.validate_environment(model="bytez/google/gemma-2b")
    assert result["keys_in_environment"] is True
    assert result["missing_keys"] == []
