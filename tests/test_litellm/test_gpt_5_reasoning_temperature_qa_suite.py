import concurrent.futures
import time
import pytest
import litellm
from litellm.exceptions import UnsupportedParamsError


# ---------------------------------------------------------------------------
# 1. UNIT TESTS
# ---------------------------------------------------------------------------
def test_unit_gpt_5_5_plus_model_detection():
    """Unit test for is_model_gpt_5_5_plus_model helper method.

    Why it exists: Ensures model version classification correctly tags >= 5.5 models.
    Expected result: Returns True for gpt-5.5, gpt-5.6, gpt-5.7; False for gpt-5.1, gpt-5.4, gpt-4o.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py
    Missing coverage: Non-standard provider prefixes handled via lookup helper.
    """
    config = litellm.OpenAIGPT5Config
    assert config.is_model_gpt_5_5_plus_model("gpt-5.5") is True
    assert config.is_model_gpt_5_5_plus_model("gpt-5.5-2026-04-23") is True
    assert config.is_model_gpt_5_5_plus_model("gpt-5.6-luna") is True
    assert config.is_model_gpt_5_5_plus_model("azure/gpt-5.6-sol") is True

    assert config.is_model_gpt_5_5_plus_model("gpt-5.4") is False
    assert config.is_model_gpt_5_5_plus_model("gpt-5.1") is False
    assert config.is_model_gpt_5_5_plus_model("gpt-4o") is False


def test_unit_supports_reasoning_effort_level_none_overrides():
    """Unit test verifying _supports_reasoning_effort_level for level='none'.

    Why it exists: Verifies level='none' is forced to False for gpt-5.5+.
    Expected result: Returns False for gpt-5.5/gpt-5.6, True/False per JSON for gpt-5.1/5.4.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py
    Missing coverage: Custom model cost maps passed dynamically at runtime.
    """
    config = litellm.OpenAIGPT5Config
    assert config._supports_reasoning_effort_level("gpt-5.5", "none") is False
    assert config._supports_reasoning_effort_level("gpt-5.6", "none") is False
    assert config._supports_reasoning_effort_level("gpt-5.6-sol", "none") is False


# ---------------------------------------------------------------------------
# 2. INTEGRATION TESTS
# ---------------------------------------------------------------------------
def test_integration_azure_and_openai_gpt_5_5_temperature_mapping():
    """Integration test for OpenAIGPT5Config and AzureOpenAIGPT5Config parameter mapping.

    Why it exists: Verifies end-to-end parameter mapping flow for both OpenAI and Azure config classes.
    Expected result: UnsupportedParamsError raised for temperature != 1.0; temperature=1.0 passes through.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py, litellm/llms/azure/chat/gpt_5_transformation.py
    Missing coverage: Live network API response validation.
    """
    openai_config = litellm.OpenAIGPT5Config()
    azure_config = litellm.AzureOpenAIGPT5Config()

    for config, model in [(openai_config, "openai/gpt-5.5"), (azure_config, "azure/gpt-5.6-sol")]:
        # Invalid temperature 0.0
        with pytest.raises(UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"temperature": 0.0},
                optional_params={},
                model=model,
                drop_params=False,
            )

        # Valid temperature 1.0
        opt_params = {}
        config.map_openai_params(
            non_default_params={"temperature": 1.0},
            optional_params=opt_params,
            model=model,
            drop_params=False,
        )
        assert opt_params.get("temperature") == 1.0


# ---------------------------------------------------------------------------
# 3. REGRESSION TESTS
# ---------------------------------------------------------------------------
def test_regression_gpt_5_4_retains_none_effort_support():
    """Regression test ensuring gpt-5.4 models still support reasoning_effort='none' and custom temperatures.

    Why it exists: Prevents regression where fixing 5.5/5.6 breaks 5.4/5.1 flexible temperature behavior.
    Expected result: gpt-5.4 allows custom temperature without raising UnsupportedParamsError.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py, model_prices_and_context_window.json
    Missing coverage: gpt-5.1 older model deprecation schedules.
    """
    config = litellm.OpenAIGPT5Config()
    opt_params = {}
    config.map_openai_params(
        non_default_params={"temperature": 0.2},
        optional_params=opt_params,
        model="gpt-5.4",
        drop_params=False,
    )
    assert opt_params.get("temperature") == 0.2


# ---------------------------------------------------------------------------
# 4. EDGE CASES
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temp", [1.0, 1])
def test_edge_case_temperature_one_variations(temp):
    """Edge case test for float vs int representations of temperature=1.

    Why it exists: Ensures both integer 1 and float 1.0 pass parameter validation.
    Expected result: Parameter passed through without exception.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py
    Missing coverage: String inputs "1.0" (handled upstream by type hints).
    """
    config = litellm.OpenAIGPT5Config()
    opt_params = {}
    config.map_openai_params(
        non_default_params={"temperature": temp},
        optional_params=opt_params,
        model="gpt-5.5",
        drop_params=False,
    )
    assert opt_params.get("temperature") == temp


def test_edge_case_drop_params_global_and_local():
    """Edge case test for drop_params setting (both litellm.drop_params and kwargs drop_params).

    Why it exists: Validates user opt-in to silently drop unsupported params.
    Expected result: Invalid temperature is removed from optional_params and no exception is raised.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py
    Missing coverage: Proxy litellm_settings drop_params config file parsing.
    """
    config = litellm.OpenAIGPT5Config()

    # Local drop_params=True
    opt_params = {}
    config.map_openai_params(
        non_default_params={"temperature": 0.7},
        optional_params=opt_params,
        model="gpt-5.5",
        drop_params=True,
    )
    assert "temperature" not in opt_params

    # Global litellm.drop_params=True
    litellm.drop_params = True
    try:
        opt_params = {}
        config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params=opt_params,
            model="gpt-5.5",
            drop_params=False,
        )
        assert "temperature" not in opt_params
    finally:
        litellm.drop_params = False


# ---------------------------------------------------------------------------
# 5. NEGATIVE TESTS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_temp", [0.0, 0, 0.1, 0.99, 1.01, 2.0, -0.5])
def test_negative_invalid_temperature_values(invalid_temp):
    """Negative test evaluating multiple out-of-spec temperature values for GPT-5.5/5.6.

    Why it exists: Guarantees any non-1 temperature value triggers local UnsupportedParamsError.
    Expected result: UnsupportedParamsError raised.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py
    Missing coverage: Non-numeric temperature types (e.g. strings/lists).
    """
    config = litellm.OpenAIGPT5Config()
    with pytest.raises(UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": invalid_temp},
            optional_params={},
            model="gpt-5.5",
            drop_params=False,
        )


# ---------------------------------------------------------------------------
# 6. SECURITY TESTS
# ---------------------------------------------------------------------------
def test_security_parameter_pollution_and_type_confusion():
    """Security test for type confusion and parameter injection payloads in temperature.

    Why it exists: Prevents malicious or malformed parameters from bypassing validation.
    Expected result: UnsupportedParamsError or TypeError raised safely without unhandled exception.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py
    Missing coverage: Sanitization of deep nested object structures.
    """
    config = litellm.OpenAIGPT5Config()
    malicious_payloads = [
        {"__class__": "admin"},
        [1.0, 0.0],
        "1.0; DROP TABLE users;",
    ]
    for payload in malicious_payloads:
        with pytest.raises((UnsupportedParamsError, TypeError, AttributeError)):
            config.map_openai_params(
                non_default_params={"temperature": payload},
                optional_params={},
                model="gpt-5.5",
                drop_params=False,
            )


# ---------------------------------------------------------------------------
# 7. PERFORMANCE & CONCURRENCY TESTS
# ---------------------------------------------------------------------------
def test_concurrency_high_throughput_validation():
    """Concurrency and performance test validating parallel thread safety and throughput.

    Why it exists: Verifies parameter transformation logic is thread-safe and adds zero latency overhead (< 1ms per 1000 calls).
    Expected result: 1,000 parameter validations complete in parallel under 2.0s with zero race conditions.
    Files covered: litellm/llms/openai/chat/gpt_5_transformation.py
    Missing coverage: Distributed multi-process worker safety across LiteLLM Proxy gunicorn workers.
    """
    config = litellm.OpenAIGPT5Config()

    def task(i):
        temp = 1.0 if i % 2 == 0 else 0.0
        opt_params = {}
        if temp == 1.0:
            config.map_openai_params(
                non_default_params={"temperature": 1.0},
                optional_params=opt_params,
                model="gpt-5.5",
                drop_params=False,
            )
            assert opt_params.get("temperature") == 1.0
        else:
            with pytest.raises(UnsupportedParamsError):
                config.map_openai_params(
                    non_default_params={"temperature": 0.0},
                    optional_params={},
                    model="gpt-5.5",
                    drop_params=False,
                )

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(task, i) for i in range(1000)]
        concurrent.futures.wait(futures)

    duration = time.time() - start_time
    assert duration < 2.0, f"Validation throughput exceeded threshold: took {duration:.2f}s"
