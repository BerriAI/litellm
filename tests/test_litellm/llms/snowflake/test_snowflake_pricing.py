import json
import os


# All 18 Snowflake models added from Table 6(b) of the Snowflake Service
# Consumption Table (REST API with Prompt Caching, Regional pricing).
SNOWFLAKE_CLAUDE_MODELS = [
    "snowflake/claude-3-7-sonnet",
    "snowflake/claude-4-opus",
    "snowflake/claude-4-sonnet",
    "snowflake/claude-haiku-4-5",
    "snowflake/claude-opus-4-5",
    "snowflake/claude-opus-4-6",
    "snowflake/claude-sonnet-4-5",
    "snowflake/claude-sonnet-4-5-long-context",
    "snowflake/claude-sonnet-4-6",
]

SNOWFLAKE_OPENAI_MODELS = [
    "snowflake/openai-gpt-4.1",
    "snowflake/openai-gpt-5",
    "snowflake/openai-gpt-5-mini",
    "snowflake/openai-gpt-5-nano",
    "snowflake/openai-gpt-5.1",
    "snowflake/openai-gpt-5.2",
    "snowflake/openai-gpt-5.4",
    "snowflake/openai-gpt-5.4-long-context",
    "snowflake/openai-o4-mini",
]

ALL_SNOWFLAKE_MODELS = SNOWFLAKE_CLAUDE_MODELS + SNOWFLAKE_OPENAI_MODELS


def _load_pricing_data():
    json_path = os.path.join(
        os.path.dirname(__file__), "../../../../model_prices_and_context_window.json"
    )
    assert os.path.exists(json_path), f"Could not find pricing JSON at {json_path}"
    with open(json_path, "r") as f:
        return json.load(f)


def test_snowflake_models_exist():
    """All 18 new Snowflake REST API models must be present in the pricing JSON."""
    data = _load_pricing_data()
    missing = [m for m in ALL_SNOWFLAKE_MODELS if m not in data]
    assert not missing, f"Missing Snowflake models: {missing}"


def test_snowflake_models_have_correct_provider():
    """Every new Snowflake model must declare litellm_provider = 'snowflake'."""
    data = _load_pricing_data()
    errors = []
    for model in ALL_SNOWFLAKE_MODELS:
        if model in data:
            provider = data[model].get("litellm_provider")
            if provider != "snowflake":
                errors.append(
                    f"{model}: litellm_provider={provider!r}, expected 'snowflake'"
                )
    assert not errors, "\n".join(errors)


def test_snowflake_models_have_positive_pricing():
    """All new Snowflake models must have positive input and output costs."""
    data = _load_pricing_data()
    errors = []
    for model in ALL_SNOWFLAKE_MODELS:
        info = data.get(model, {})
        for field in ("input_cost_per_token", "output_cost_per_token"):
            val = info.get(field)
            if val is None:
                errors.append(f"{model}: missing {field}")
            elif val <= 0:
                errors.append(f"{model}: {field}={val} is not positive")
    assert not errors, "\n".join(errors)


def test_snowflake_claude_models_have_prompt_caching_fields():
    """Claude models on Snowflake support prompt caching and must include both
    cache_creation_input_token_cost and cache_read_input_token_cost."""
    data = _load_pricing_data()
    errors = []
    for model in SNOWFLAKE_CLAUDE_MODELS:
        info = data.get(model, {})
        for field in ("cache_creation_input_token_cost", "cache_read_input_token_cost"):
            val = info.get(field)
            if val is None:
                errors.append(f"{model}: missing {field}")
            elif val <= 0:
                errors.append(f"{model}: {field}={val} is not positive")
        if not info.get("supports_prompt_caching"):
            errors.append(f"{model}: supports_prompt_caching should be True")
    assert not errors, "\n".join(errors)


def test_snowflake_openai_models_have_cache_read_but_no_cache_write():
    """OpenAI models on Snowflake (Azure) have cache read pricing but no cache
    write cost (Table 6b shows '-' for cache write on OpenAI models)."""
    data = _load_pricing_data()
    errors = []
    for model in SNOWFLAKE_OPENAI_MODELS:
        info = data.get(model, {})
        # Must have cache read cost
        val = info.get("cache_read_input_token_cost")
        if val is None:
            errors.append(f"{model}: missing cache_read_input_token_cost")
        elif val <= 0:
            errors.append(f"{model}: cache_read_input_token_cost={val} is not positive")
        # Must NOT have cache creation cost
        if "cache_creation_input_token_cost" in info:
            errors.append(
                f"{model}: unexpected cache_creation_input_token_cost "
                f"(OpenAI models have no cache write pricing)"
            )
    assert not errors, "\n".join(errors)


def test_snowflake_models_have_context_window():
    """All new Snowflake models must define max_input_tokens and max_output_tokens."""
    data = _load_pricing_data()
    errors = []
    for model in ALL_SNOWFLAKE_MODELS:
        info = data.get(model, {})
        for field in ("max_input_tokens", "max_output_tokens", "max_tokens"):
            if info.get(field) is None:
                errors.append(f"{model}: missing {field}")
    assert not errors, "\n".join(errors)
