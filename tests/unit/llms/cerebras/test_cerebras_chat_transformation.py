import pytest

import litellm
from litellm.llms.cerebras.chat import CerebrasConfig


def test_max_retries_in_supported_params() -> None:
    config = CerebrasConfig()
    params = config.get_supported_openai_params(model="llama-3.3-70b")
    assert "max_retries" in params, (
        f"max_retries must be in CerebrasConfig.get_supported_openai_params(); got: {params!r}"
    )


def test_extra_headers_in_supported_params() -> None:
    config = CerebrasConfig()
    params = config.get_supported_openai_params(model="llama-3.3-70b")
    assert "extra_headers" in params, (
        f"extra_headers must be in CerebrasConfig.get_supported_openai_params(); got: {params!r}"
    )


def test_core_openai_params_still_supported() -> None:
    config = CerebrasConfig()
    params = config.get_supported_openai_params(model="llama-3.3-70b")
    for expected in (
        "max_tokens",
        "max_completion_tokens",
        "response_format",
        "seed",
        "stop",
        "stream",
        "temperature",
        "top_p",
        "tool_choice",
        "tools",
        "user",
    ):
        assert expected in params, f"{expected!r} unexpectedly missing from Cerebras supported params: {params!r}"


def test_map_openai_params_preserves_max_retries() -> None:
    config = CerebrasConfig()
    result = config.map_openai_params(
        non_default_params={"max_retries": 0, "temperature": 0.7},
        optional_params={},
        model="llama-3.3-70b",
        drop_params=False,
    )
    assert result.get("max_retries") == 0, f"map_openai_params must preserve max_retries=0; got: {result!r}"
    assert result.get("temperature") == 0.7


def test_map_openai_params_preserves_max_retries_zero_falsy() -> None:
    config = CerebrasConfig()
    result = config.map_openai_params(
        non_default_params={"max_retries": 0},
        optional_params={},
        model="llama-3.3-70b",
        drop_params=False,
    )
    assert "max_retries" in result and result["max_retries"] == 0, (
        f"max_retries=0 (falsy) must not be silently omitted; got: {result!r}"
    )


@pytest.mark.parametrize(
    ("model", "expected_input_cost", "expected_output_cost"),
    [
        ("cerebras/llama-3.1-8b", 1e-07, 1e-07),
        ("cerebras/llama-3.1-70b", 6e-07, 6e-07),
        ("cerebras/llama-3.3-70b", 8.5e-07, 1.2e-06),
    ],
)
def test_hyphenated_cerebras_llama_model_pricing_and_info(
    model: str,
    expected_input_cost: float,
    expected_output_cost: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    info = litellm.get_model_info(model=model)
    assert info.get("litellm_provider") == "cerebras"
    assert info.get("input_cost_per_token") == expected_input_cost
    assert info.get("output_cost_per_token") == expected_output_cost
    assert info.get("max_tokens") == 128000
    assert info.get("supports_function_calling") is True

    prompt_tokens = 1000
    completion_tokens = 500
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    assert prompt_cost == prompt_tokens * expected_input_cost
    assert completion_cost == completion_tokens * expected_output_cost

