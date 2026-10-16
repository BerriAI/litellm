import litellm
from litellm.llms.codestral.completion.transformation import (
    CodestralTextCompletionConfig,
)


def test_codestral_supported_openai_params_includes_min_tokens():
    config = CodestralTextCompletionConfig()
    supported_params = config.get_supported_openai_params(model="codestral-latest")
    assert "min_tokens" in supported_params
    assert "max_tokens" in supported_params
    assert "suffix" in supported_params


def test_codestral_map_openai_params_min_tokens():
    config = CodestralTextCompletionConfig()
    mapped = config.map_openai_params(
        non_default_params={"min_tokens": 10},
        optional_params={},
        model="codestral-latest",
        drop_params=False,
    )
    assert mapped == {"min_tokens": 10}


def test_get_supported_openai_params_for_text_completion_codestral():
    supported = litellm.get_supported_openai_params(
        model="codestral/codestral-latest",
        custom_llm_provider="text-completion-codestral",
    )
    assert "min_tokens" in supported
