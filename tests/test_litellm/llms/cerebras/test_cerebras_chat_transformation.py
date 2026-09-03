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
