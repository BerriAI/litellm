"""Tests for exponential cooldown backoff."""
import pytest
from unittest.mock import MagicMock, patch

from litellm.router_utils.cooldown_cache import CooldownCache
from litellm.caching.caching import DualCache


@pytest.fixture
def cooldown_cache():
    cache = DualCache()
    return CooldownCache(cache=cache, default_cooldown_time=1.0)


class TestExponentialCooldown:
    def test_first_failure_uses_base_cooldown(self, cooldown_cache):
        """First failure: cooldown = base (1s)."""
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldowns = cooldown_cache.get_active_cooldowns(
            model_ids=["dep-1"], parent_otel_span=None
        )
        assert len(cooldowns) == 1
        assert cooldowns[0][1]["cooldown_time"] == 1.0

    def test_second_failure_doubles_cooldown(self, cooldown_cache):
        """Second consecutive failure: cooldown = 2s."""
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldowns = cooldown_cache.get_active_cooldowns(
            model_ids=["dep-1"], parent_otel_span=None
        )
        assert len(cooldowns) == 1
        assert cooldowns[0][1]["cooldown_time"] == 2.0

    def test_cooldown_caps_at_60_seconds(self, cooldown_cache):
        """Cooldown should cap at 60 seconds."""
        for _ in range(7):
            cooldown_cache.add_deployment_to_cooldown(
                model_id="dep-1",
                original_exception=Exception("error"),
                exception_status=500,
                cooldown_time=None,
            )
        cooldowns = cooldown_cache.get_active_cooldowns(
            model_ids=["dep-1"], parent_otel_span=None
        )
        assert cooldowns[0][1]["cooldown_time"] == 60.0

    def test_reset_failure_count_on_success(self, cooldown_cache):
        """After success, failure count resets so next failure uses base cooldown."""
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldown_cache.reset_deployment_failure_count(model_id="dep-1")
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldowns = cooldown_cache.get_active_cooldowns(
            model_ids=["dep-1"], parent_otel_span=None
        )
        assert cooldowns[0][1]["cooldown_time"] == 1.0

    def test_independent_failure_counts_per_deployment(self, cooldown_cache):
        """Each deployment tracks failures independently."""
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-2",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=None,
        )
        cooldowns = cooldown_cache.get_active_cooldowns(
            model_ids=["dep-1", "dep-2"], parent_otel_span=None
        )
        cooldown_map = {cd[0]: cd[1]["cooldown_time"] for cd in cooldowns}
        assert cooldown_map["dep-1"] == 2.0
        assert cooldown_map["dep-2"] == 1.0

    def test_explicit_cooldown_time_overrides_exponential(self, cooldown_cache):
        """When cooldown_time is explicitly passed, use it instead of exponential."""
        cooldown_cache.add_deployment_to_cooldown(
            model_id="dep-1",
            original_exception=Exception("error"),
            exception_status=500,
            cooldown_time=10.0,
        )
        cooldowns = cooldown_cache.get_active_cooldowns(
            model_ids=["dep-1"], parent_otel_span=None
        )
        assert cooldowns[0][1]["cooldown_time"] == 10.0
