"""Freshness is one predicate, derived from the provider prompt-cache TTL."""

import pytest

from litellm.router_strategy.complexity_router.cache_warming.types import (
    PROVIDER_PROMPT_CACHE_TTL_SECONDS,
    is_cache_fresh,
    needs_rewarming,
)


@pytest.mark.parametrize(
    "age,fresh",
    [(0, True), (PROVIDER_PROMPT_CACHE_TTL_SECONDS - 1, True), (PROVIDER_PROMPT_CACHE_TTL_SECONDS, False), (601, False)],
)
def test_freshness_is_the_provider_ttl_and_nothing_else(age, fresh):
    """The router's warm-aware pick and the refresher's due-model calculation both read this, so a model can
    never be preferred as warm by one while the other treats it as stale. An operator's idle_timeout_seconds
    of 600 must not make a 400-second-old prefix look warm."""
    assert is_cache_fresh(1000.0 - age, 1000.0) is fresh


@pytest.mark.parametrize("refresh_interval,age,due", [(120, 119, False), (120, 120, True), (3000, 240, True)])
def test_rewarming_never_waits_past_the_provider_ttl(refresh_interval, age, due):
    """A short interval governs on its own; an interval longer than the TTL is capped by the TTL less the tick
    slack, so it cannot open a window where the pick still believes a model is warm."""
    assert needs_rewarming(1000.0 - age, 1000.0, refresh_interval) is due


def test_attribution_covers_every_identity_field_the_proxy_stamps():
    """A replay is authorized against a principal rebuilt from attribution, so any identity id the proxy
    stamps and the auth gates resolve by must be captured. If litellm adds one, this fails rather than
    silently leaving that gate unbound for keyless callers."""
    from litellm.types.utils import StandardLoggingUserAPIKeyMetadata

    from litellm.router_strategy.complexity_router.cache_warming.types import CacheWarmingAttribution

    upstream_ids = {
        field
        for field in StandardLoggingUserAPIKeyMetadata.__annotations__
        if field.endswith("_id") and field.startswith("user_api_key_")
    }
    assert upstream_ids, "upstream identity metadata shape changed"
    assert upstream_ids <= set(CacheWarmingAttribution.model_fields), (
        f"attribution is missing identity fields the proxy stamps: {upstream_ids - set(CacheWarmingAttribution.model_fields)}"
    )
