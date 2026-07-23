import litellm


def test_validate_environment_morph_missing(monkeypatch):
    monkeypatch.delenv("MORPH_API_KEY", raising=False)
    result = litellm.validate_environment("morph/morph-v3-fast")
    assert result["keys_in_environment"] is False
    assert "MORPH_API_KEY" in result["missing_keys"]


def test_validate_environment_morph_present(monkeypatch):
    monkeypatch.setenv("MORPH_API_KEY", "sk-test")
    result = litellm.validate_environment("morph/morph-v3-fast")
    assert result["keys_in_environment"] is True
    assert result["missing_keys"] == []
