from datetime import datetime
from unittest.mock import patch

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.types.utils import ModelResponse, Usage


def _kwargs(model_info: object) -> dict:
    return {
        "litellm_params": {
            "metadata": {"model_group": "gpt-3.5-turbo"},
            "model_info": model_info,
        },
        "response_cost": 0.01,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_info", [None, {}])
async def test_log_success_event_without_deployment_id_does_not_raise(model_info):
    """
    litellm_params carries an explicit `model_info: None` for calls that aren't routed
    through a router deployment; the handler must no-op instead of blowing up on
    `None.get("id")` and spamming exception logs (issue #35459)
    """
    handler = LowestCostLoggingHandler(router_cache=DualCache(), routing_args={})
    response_obj = ModelResponse(usage=Usage(prompt_tokens=25, completion_tokens=25, total_tokens=50))
    now = datetime.now()

    with patch("litellm.router_strategy.lowest_cost.verbose_logger.exception") as mock_exception:
        handler.log_success_event(_kwargs(model_info), response_obj, now, now)
        await handler.async_log_success_event(_kwargs(model_info), response_obj, now, now)

    mock_exception.assert_not_called()
    assert handler.router_cache.get_cache("gpt-3.5-turbo_map") is None


@pytest.mark.asyncio
async def test_log_success_event_records_cost_for_routed_deployment():
    handler = LowestCostLoggingHandler(router_cache=DualCache(), routing_args={})
    response_obj = ModelResponse(usage=Usage(prompt_tokens=25, completion_tokens=25, total_tokens=50))
    now = datetime.now()

    await handler.async_log_success_event(_kwargs({"id": "deployment-1"}), response_obj, now, now)

    cost_map = handler.router_cache.get_cache("gpt-3.5-turbo_map")
    assert cost_map is not None
    assert "deployment-1" in cost_map
