import pytest

import litellm


@pytest.mark.parametrize("model", ["mistral", "openai", "anthropic", "groq", "deepseek"])
def test_get_llm_provider_bare_provider_name_raises_helpful_error(model):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.get_llm_provider(model=model)
    message = str(exc_info.value)
    assert "LLM Provider NOT provided" in message
    assert "list index out of range" not in message
