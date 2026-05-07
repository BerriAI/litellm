"""
Performance benchmarks for litellm core operations.

These benchmarks measure the performance of frequently called functions
in the litellm hot path: token counting, model info lookup, provider
resolution, and cost calculation.
"""

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.token_counter import token_counter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_MESSAGES = [{"role": "user", "content": "Hello, how are you?"}]

MULTI_TURN_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {
        "role": "assistant",
        "content": "The capital of France is Paris. It is known as the City of Light.",
    },
    {"role": "user", "content": "Tell me more about Paris."},
    {
        "role": "assistant",
        "content": (
            "Paris is the capital and most populous city of France. "
            "With an estimated population of 2,165,423 in 2019, it is the "
            "centre of the Ile-de-France region. The city is a major European "
            "cultural and commercial centre."
        ),
    },
    {"role": "user", "content": "What are the top tourist attractions?"},
]

LONG_CONTENT_MESSAGE = [
    {
        "role": "user",
        "content": "Explain the following concept in detail: " + "word " * 500,
    }
]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Token counting benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_token_counter_simple_message():
    """Benchmark token counting for a single short message."""
    token_counter(model="gpt-4o", messages=SIMPLE_MESSAGES)


@pytest.mark.benchmark
def test_token_counter_multi_turn():
    """Benchmark token counting for a multi-turn conversation."""
    token_counter(model="gpt-4o", messages=MULTI_TURN_MESSAGES)


@pytest.mark.benchmark
def test_token_counter_long_content():
    """Benchmark token counting for a message with long content."""
    token_counter(model="gpt-4o", messages=LONG_CONTENT_MESSAGE)


@pytest.mark.benchmark
def test_token_counter_with_tools():
    """Benchmark token counting with tool definitions."""
    token_counter(
        model="gpt-4o",
        messages=SIMPLE_MESSAGES,
        tools=TOOL_DEFINITIONS,
    )


@pytest.mark.benchmark
def test_token_counter_raw_text():
    """Benchmark token counting for raw text input."""
    token_counter(model="gpt-4o", text="The quick brown fox jumps over the lazy dog.")


# ---------------------------------------------------------------------------
# Model info lookup benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_get_model_info_openai():
    """Benchmark model info lookup for an OpenAI model."""
    litellm.get_model_info("gpt-4o")


@pytest.mark.benchmark
def test_get_model_info_anthropic():
    """Benchmark model info lookup for an Anthropic model."""
    litellm.get_model_info("claude-sonnet-4-20250514")


@pytest.mark.benchmark
def test_get_model_info_with_provider():
    """Benchmark model info lookup with an explicit provider prefix."""
    litellm.get_model_info("openai/gpt-4o", custom_llm_provider="openai")


# ---------------------------------------------------------------------------
# Provider resolution benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_get_llm_provider_openai():
    """Benchmark LLM provider resolution for OpenAI."""
    get_llm_provider(model="gpt-4o")


@pytest.mark.benchmark
def test_get_llm_provider_anthropic():
    """Benchmark LLM provider resolution for Anthropic."""
    get_llm_provider(model="claude-sonnet-4-20250514")


@pytest.mark.benchmark
def test_get_llm_provider_with_prefix():
    """Benchmark LLM provider resolution with provider prefix."""
    get_llm_provider(model="openai/gpt-4o")


@pytest.mark.benchmark
def test_get_llm_provider_azure():
    """Benchmark LLM provider resolution for Azure."""
    get_llm_provider(
        model="azure/gpt-4o",
        api_base="https://my-endpoint.openai.azure.com",
    )


# ---------------------------------------------------------------------------
# Cost calculation benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_cost_per_token_openai():
    """Benchmark cost-per-token calculation for OpenAI models."""
    litellm.cost_per_token(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
    )


@pytest.mark.benchmark
def test_cost_per_token_anthropic():
    """Benchmark cost-per-token calculation for Anthropic models."""
    litellm.cost_per_token(
        model="claude-sonnet-4-20250514",
        prompt_tokens=1000,
        completion_tokens=500,
    )


# ---------------------------------------------------------------------------
# Model cost key resolution benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_get_model_cost_key_exact_match():
    """Benchmark model cost key lookup with an exact match."""
    litellm.utils._get_model_cost_key("gpt-4o")


@pytest.mark.benchmark
def test_get_model_cost_key_case_insensitive():
    """Benchmark model cost key lookup with case-insensitive fallback."""
    litellm.utils._get_model_cost_key("GPT-4o")
