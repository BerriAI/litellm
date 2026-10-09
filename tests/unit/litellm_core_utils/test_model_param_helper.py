

from litellm.litellm_core_utils.model_param_helper import ModelParamHelper


def test_get_all_llm_api_params_is_correct():
    """The cached result must equal a fresh, uncached computation."""
    cached = ModelParamHelper.get_all_llm_api_params()
    uncached = ModelParamHelper.get_all_llm_api_params.__wrapped__()
    assert cached == uncached
    assert {"model", "temperature", "stream"} <= cached
    assert "metadata" not in cached  # excluded via _get_exclude_kwargs


def test_get_all_llm_api_params_is_memoized():
    """Regression: the param set is static and is rebuilt on every request via
    the cache-key and spend-logging paths, so it must be memoized. Without the
    cache each call returns a freshly built set (a different object)."""
    first = ModelParamHelper.get_all_llm_api_params()
    second = ModelParamHelper.get_all_llm_api_params()
    assert first is second


def test_langfuse_model_parameters_no_secret_leakage():
    optional_params_with_secrets = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100,
        "stream": True,
        "api_key": "sk-secret-key-12345",
        "api_base": "https://my-private-endpoint.com",
        "secret_fields": {"raw_headers": {"Authorization": "Bearer sk-super-secret"}},
        "authorization": "Bearer sk-another-secret",
        "headers": {"X-Api-Key": "secret-header-value"},
    }

    sanitized = ModelParamHelper.get_standard_logging_model_parameters(optional_params_with_secrets)

    assert sanitized["temperature"] == 0.7
    assert sanitized["top_p"] == 0.9
    assert sanitized["max_tokens"] == 100
    assert sanitized["stream"] is True

    assert "api_key" not in sanitized
    assert "api_base" not in sanitized
    assert "secret_fields" not in sanitized
    assert "authorization" not in sanitized
    assert "headers" not in sanitized
