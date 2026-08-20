import pytest

from litellm.llms.cerebras.chat import CerebrasConfig


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_completion_tokens", 64),
        ("max_tokens", 32),
        ("parallel_tool_calls", False),
        ("logprobs", True),
        ("top_logprobs", 3),
        ("frequency_penalty", 0.2),
        ("presence_penalty", 0.3),
        ("logit_bias", {"42": -1}),
        ("service_tier", "default"),
        ("prompt_cache_key", "conversation-1"),
        ("prediction", {"type": "content", "content": "expected"}),
    ],
)
def test_cerebras_preserves_supported_parameters(parameter: str, value: object) -> None:
    config = CerebrasConfig()

    mapped = config.map_openai_params(
        non_default_params={parameter: value},
        optional_params={},
        model="gpt-oss-120b",
        drop_params=False,
    )

    assert mapped == {parameter: value}


def test_cerebras_does_not_alias_max_completion_tokens() -> None:
    config = CerebrasConfig()

    mapped = config.map_openai_params(
        non_default_params={"max_completion_tokens": 64},
        optional_params={},
        model="gpt-oss-120b",
        drop_params=False,
    )

    assert mapped["max_completion_tokens"] == 64
    assert "max_tokens" not in mapped
