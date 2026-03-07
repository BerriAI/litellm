from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base


def test_get_api_base_rejects_non_string_api_base():
    """When api_base is in dict but not a string, fast path should be skipped."""
    result = get_api_base(model="gpt-3.5-turbo", optional_params={"api_base": 123})
    assert result is None or isinstance(result, str)


def test_get_api_base_fast_path_returns_string():
    """When api_base is a valid string in dict, fast path should return it directly."""
    result = get_api_base(
        model="gpt-3.5-turbo",
        optional_params={"api_base": "https://my-proxy.example.com"},
    )
    assert result == "https://my-proxy.example.com"
