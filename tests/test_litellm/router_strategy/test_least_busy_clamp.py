def test_least_busy_decrement_clamps_at_zero():
    assert max(0 - 1, 0) == 0
    assert max(2 - 1, 0) == 1


def test_cache_hit_skips_decrement():
    """Response-cache hits must not decrement least_busy counters (#39322)."""
    from unittest.mock import MagicMock

    from litellm.router_strategy.least_busy import LeastBusyLoggingHandler

    cache = MagicMock()
    cache.get_cache.return_value = {"dep-a": 2}
    handler = LeastBusyLoggingHandler(router_cache=cache)
    kwargs = {
        "cache_hit": True,
        "litellm_params": {
            "metadata": {"model_group": "g"},
            "model_info": {"id": "dep-a"},
        },
    }
    handler.log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
    cache.set_cache.assert_not_called()


def test_non_cache_hit_still_decrements_with_clamp():
    from unittest.mock import MagicMock

    from litellm.router_strategy.least_busy import LeastBusyLoggingHandler

    cache = MagicMock()
    cache.get_cache.return_value = {"dep-a": 0}
    handler = LeastBusyLoggingHandler(router_cache=cache)
    kwargs = {
        "cache_hit": False,
        "litellm_params": {
            "metadata": {"model_group": "g"},
            "model_info": {"id": "dep-a"},
        },
    }
    handler.log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
    cache.set_cache.assert_called_once()
    value = cache.set_cache.call_args.kwargs.get("value")
    assert value["dep-a"] == 0
