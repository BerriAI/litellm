"""
Performance benchmarks for litellm core operations.

These benchmarks measure the performance of frequently called functions
in the litellm hot path: token counting, model info lookup, provider
resolution, and cost calculation.
"""

import io
import tempfile
import threading
import wave

import pytest

import litellm
from litellm.litellm_core_utils.audio_utils.utils import calculate_request_duration
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.thread_pool_executor import executor
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


# ---------------------------------------------------------------------------
# Audio duration extraction
# ---------------------------------------------------------------------------


def _benchmark_wav(duration_seconds: float, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(sample_rate * duration_seconds))
    return buffer.getvalue()


def _spooled_upload(content: bytes) -> tempfile.SpooledTemporaryFile:
    """A starlette-shaped upload. A BytesIO over an existing bytes object shares
    its buffer, so reading it costs nothing and would understate the real work."""
    handle = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
    handle.write(content)
    handle.seek(0)
    return handle


SHORT_UPLOAD = _spooled_upload(_benchmark_wav(5.0))
LONG_UPLOAD = _spooled_upload(_benchmark_wav(600.0))
UNREADABLE_UPLOAD = _spooled_upload(b"\x00\x00\x00\x20ftypM4A " + b"\x11" * (2 * 1024 * 1024))


@pytest.mark.benchmark
def test_audio_duration_short_wav():
    """Duration read for a 5s WAV, the common transcription upload."""
    calculate_request_duration(SHORT_UPLOAD)


@pytest.mark.benchmark
def test_audio_duration_long_wav():
    """Duration read for 10 minutes of WAV, ~19 MB, near the provider size cap."""
    calculate_request_duration(LONG_UPLOAD)


@pytest.mark.benchmark
def test_audio_duration_unreadable_container():
    """Measures the failure path taken by any upload libsndfile declines to
    decode, before the byte-count ceiling picks it up."""
    calculate_request_duration(UNREADABLE_UPLOAD)


# ---------------------------------------------------------------------------
# Measurement hermeticity guard
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_logging_executor_runs_inline():
    """Guard that the shared logging executor runs submissions inline.

    Deferred submissions execute on worker threads, and callgrind attributes
    their instructions to whichever benchmark's measurement window is open when
    the valgrind scheduler resumes them, making results nondeterministic.
    """
    future = executor.submit(threading.get_ident)
    assert future.done()
    assert future.result() == threading.get_ident()
