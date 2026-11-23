import asyncio
import importlib
from typing import List
from unittest.mock import MagicMock

import pytest
import litellm
from litellm.router_api import RouterSettings
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.asyncio
async def test_async_raise_no_deployment_exception():
    """
    Test that async_raise_no_deployment_exception returns a RouterRateLimitError
    with cooldown_list containing just IDs (not tuples with debug info).
    """
    from unittest.mock import patch

    from litellm.router_utils.handle_error import async_raise_no_deployment_exception
    from litellm.types.router import RouterRateLimitError

    # A routing context: the group's deployment ids, their cooldown store and the router settings
    ctx = MagicMock()
    ctx.registry.return_value.names.return_value = ["deployment-1", "deployment-2"]
    ctx.health.return_value.cooldowns.get_min_cooldown.return_value = 30.0
    ctx.settings.return_value = RouterSettings(enable_pre_call_checks=True)

    # Mock the _async_get_cooldown_deployments_with_debug_info function
    # It should return a list of tuples where each tuple contains (model_id, debug_info)
    mock_cooldown_list = [
        ("deployment-1", {"error": "rate_limit", "time": "2024-01-01"}),
        ("deployment-2", {"error": "server_error", "time": "2024-01-01"}),
        ("deployment-3", {"error": "timeout", "time": "2024-01-01"}),
    ]

    with patch(
        "litellm.router_utils.handle_error.async_get_cooldown_deployments_with_debug_info",
        return_value=mock_cooldown_list,
    ):
        # Call the function
        result = await async_raise_no_deployment_exception(
            ctx=ctx,
            model="gpt-5-mini",
            parent_otel_span=None,
        )

    # Assert that the function returns a RouterRateLimitError
    assert isinstance(result, RouterRateLimitError)

    # Assert that the error has the correct properties
    assert result.model == "gpt-5-mini"
    assert result.cooldown_time == 30.0
    assert result.enable_pre_call_checks is True

    # Assert that cooldown_list contains only IDs (extracted from tuples)
    expected_cooldown_list = ["deployment-1", "deployment-2", "deployment-3"]
    assert result.cooldown_list == expected_cooldown_list

    # Verify that cooldown_list contains only strings (IDs), not tuples
    for item in result.cooldown_list:
        assert isinstance(item, str), f"Expected string ID, got {type(item)}: {item}"

    # Verify mock calls
    ctx.registry().names.assert_called_once_with("gpt-5-mini", kind="ids")
    ctx.health().cooldowns.get_min_cooldown.assert_called_once_with(
        model_ids=["deployment-1", "deployment-2"], parent_otel_span=None
    )


@pytest.mark.asyncio
async def test_async_raise_no_deployment_exception_empty_cooldown_list():
    """
    Test that async_raise_no_deployment_exception handles empty cooldown list correctly.
    """
    from unittest.mock import patch

    from litellm.router_utils.handle_error import async_raise_no_deployment_exception
    from litellm.types.router import RouterRateLimitError

    # A routing context: the group's deployment ids, their cooldown store and the router settings
    ctx = MagicMock()
    ctx.registry.return_value.names.return_value = ["deployment-1", "deployment-2"]
    ctx.health.return_value.cooldowns.get_min_cooldown.return_value = 15.0
    ctx.settings.return_value = RouterSettings(enable_pre_call_checks=False)

    # Mock empty cooldown list
    mock_cooldown_list: List = []

    with patch(
        "litellm.router_utils.handle_error.async_get_cooldown_deployments_with_debug_info",
        return_value=mock_cooldown_list,
    ):
        # Call the function
        result = await async_raise_no_deployment_exception(
            ctx=ctx,
            model="claude-3-sonnet",
            parent_otel_span=None,
        )

    # Assert that the function returns a RouterRateLimitError
    assert isinstance(result, RouterRateLimitError)

    # Assert that the error has the correct properties
    assert result.model == "claude-3-sonnet"
    assert result.cooldown_time == 15.0
    assert result.enable_pre_call_checks is False

    # Assert that cooldown_list is an empty list when no cooldowns exist
    assert result.cooldown_list == []
    assert isinstance(result.cooldown_list, list)


@pytest.mark.asyncio
async def test_async_raise_no_deployment_exception_none_cooldown_list():
    """
    Test that async_raise_no_deployment_exception handles None cooldown list correctly.
    Note: In practice, _async_get_cooldown_deployments_with_debug_info should never return None
    based on the implementation, but this tests defensive programming.
    """
    from unittest.mock import patch

    from litellm.router_utils.handle_error import async_raise_no_deployment_exception
    from litellm.types.router import RouterRateLimitError

    # A routing context: the group's deployment ids, their cooldown store and the router settings
    ctx = MagicMock()
    ctx.registry.return_value.names.return_value = []
    ctx.health.return_value.cooldowns.get_min_cooldown.return_value = 45.0
    ctx.settings.return_value = RouterSettings(enable_pre_call_checks=True)

    # Mock None cooldown list (though this shouldn't happen in practice)
    mock_cooldown_list = None

    with patch(
        "litellm.router_utils.handle_error.async_get_cooldown_deployments_with_debug_info",
        return_value=mock_cooldown_list,
    ):
        # After the defensive fix, this should handle None gracefully and return empty list
        result = await async_raise_no_deployment_exception(
            ctx=ctx,
            model="gpt-5.5",
            parent_otel_span=None,
        )

    # Assert that the function returns a RouterRateLimitError
    assert isinstance(result, RouterRateLimitError)

    # Assert that the error has the correct properties
    assert result.model == "gpt-5.5"
    assert result.cooldown_time == 45.0
    assert result.enable_pre_call_checks is True

    # Assert that cooldown_list is an empty list when cooldown_list is None
    assert result.cooldown_list == []
    assert isinstance(result.cooldown_list, list)


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception:
        pass
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)
