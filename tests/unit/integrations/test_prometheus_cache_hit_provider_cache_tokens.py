"""
Regression test for BerriAI/litellm#51909.

A response-cache hit replays the usage object stored with the cached response.
The provider prompt-cache counters describe tokens the provider actually read
from, or wrote to, its prompt cache, so they must only move when a provider
request happened. Before the fix, every response-cache hit added the stored
``prompt_tokens_details.cached_tokens`` to
``litellm_provider_cache_read_input_tokens_metric`` and
``litellm_input_cached_tokens_metric`` again, doubling both.

Run with:
    uv run pytest tests/unit/integrations/test_prometheus_cache_hit_provider_cache_tokens.py -v
"""

from typing import Final

import pytest
from prometheus_client import REGISTRY, Counter

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues

CACHED_PROMPT_TOKENS: Final = 25
CACHE_WRITE_TOKENS: Final = 10
PROMPT_TOKENS: Final = 80
COMPLETION_TOKENS: Final = 20


def _clear_prometheus_registry() -> None:
    for collector in tuple(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]  # test registry reset
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def isolated_registry():
    _clear_prometheus_registry()
    yield
    _clear_prometheus_registry()


@pytest.fixture
def logger() -> PrometheusLogger:
    return PrometheusLogger()


@pytest.fixture
def enum_values() -> UserAPIKeyLabelValues:
    return UserAPIKeyLabelValues(
        end_user="cache-repro-user",
        hashed_api_key="cache-repro-key",
        api_key_alias="cache-repro-alias",
        team="cache-repro-team",
        team_alias="cache-repro-team-alias",
        user="cache-repro-user-id",
        model="openai/cache-repro",
        requested_model="openai/cache-repro",
        model_group="openai/cache-repro",
        status_code="200",
    )


def _total(counter: Counter) -> float:
    return sum(
        sample.value
        for family in counter.collect()
        for sample in family.samples
        if sample.name == f"{family.name}_total"
    )


def _payload(*, cache_hit: bool | None, usage_object: dict) -> dict:
    return {
        "cache_hit": cache_hit,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "model_group": "openai/cache-repro",
        "request_tags": [],
        "metadata": {"usage_object": usage_object},
    }


def _log_cache_metrics(
    logger: PrometheusLogger,
    enum_values: UserAPIKeyLabelValues,
    *,
    cache_hit: bool | None,
    usage_object: dict,
) -> None:
    """Emit the two metric families that read the stored usage object."""
    payload: Final = _payload(cache_hit=cache_hit, usage_object=usage_object)
    logger._increment_cache_metrics(standard_logging_payload=payload, enum_values=enum_values)
    logger._increment_token_metrics(
        standard_logging_payload=payload,
        end_user_id=None,
        user_api_key=None,
        user_api_key_alias=None,
        model="openai/cache-repro",
        user_api_team=None,
        user_api_team_alias=None,
        user_id=None,
        enum_values=enum_values,
    )


def test_response_cache_hit_does_not_replay_provider_cache_read_tokens(logger, enum_values):
    usage_object: Final = {
        "prompt_tokens": PROMPT_TOKENS,
        "prompt_tokens_details": {"cached_tokens": CACHED_PROMPT_TOKENS},
    }

    _log_cache_metrics(logger, enum_values, cache_hit=False, usage_object=usage_object)
    _log_cache_metrics(logger, enum_values, cache_hit=True, usage_object=usage_object)

    assert _total(logger.litellm_provider_cache_read_input_tokens_metric) == CACHED_PROMPT_TOKENS
    assert _total(logger.litellm_input_cached_tokens_metric) == CACHED_PROMPT_TOKENS

    # The response-cache side of the accounting still counts the served hit.
    assert _total(logger.litellm_cache_hits_metric) == 1.0
    assert _total(logger.litellm_cached_tokens_metric) == PROMPT_TOKENS + COMPLETION_TOKENS


def test_response_cache_hit_does_not_replay_provider_cache_write_tokens(logger, enum_values):
    usage_object: Final = {
        "prompt_tokens": PROMPT_TOKENS,
        "cache_creation_input_tokens": CACHE_WRITE_TOKENS,
        "prompt_tokens_details": {"cache_write_tokens": CACHE_WRITE_TOKENS},
    }

    _log_cache_metrics(logger, enum_values, cache_hit=False, usage_object=usage_object)
    _log_cache_metrics(logger, enum_values, cache_hit=True, usage_object=usage_object)

    assert _total(logger.litellm_provider_cache_creation_input_tokens_metric) == CACHE_WRITE_TOKENS
    assert _total(logger.litellm_input_cache_creation_tokens_metric) == CACHE_WRITE_TOKENS


def test_uncached_requests_still_emit_provider_cache_tokens(logger, enum_values):
    """cache_hit None means no response-cache lookup happened, not a replay."""
    usage_object: Final = {
        "prompt_tokens": PROMPT_TOKENS,
        "prompt_tokens_details": {"cached_tokens": CACHED_PROMPT_TOKENS},
    }

    _log_cache_metrics(logger, enum_values, cache_hit=None, usage_object=usage_object)

    assert _total(logger.litellm_provider_cache_read_input_tokens_metric) == CACHED_PROMPT_TOKENS
    assert _total(logger.litellm_input_cached_tokens_metric) == CACHED_PROMPT_TOKENS
    assert _total(logger.litellm_cache_hits_metric) == 0.0
    assert _total(logger.litellm_cache_misses_metric) == 0.0
