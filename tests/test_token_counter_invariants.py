import pytest

def normalize_model_name(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower().replace("openai/", "").replace("anthropic/", "")

def test_empty_and_whitespace_model_normalization():
    assert normalize_model_name("") == ""
    assert normalize_model_name("   ") == ""
    assert normalize_model_name("openai/gpt-4o") == "gpt-4o"
    assert normalize_model_name("Anthropic/claude-3-5-sonnet-20241022") == "claude-3-5-sonnet-20241022"

def test_model_name_casing_invariance():
    assert normalize_model_name("GPT-4-TURBO") == "gpt-4-turbo"
    assert normalize_model_name("text-embedding-3-small") == "text-embedding-3-small"
