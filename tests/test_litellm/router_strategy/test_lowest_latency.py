import json
from datetime import datetime, timedelta

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_non_chat_latency_is_json_serializable(sync_mode: bool) -> None:
    cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=cache)
    model_group = "embedding-group"
    deployment_id = "embedding-deployment"
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": model_group},
            "model_info": {"id": deployment_id},
        }
    }
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1.25)
    response = litellm.EmbeddingResponse()

    if sync_mode:
        handler.log_success_event(kwargs, response, start_time, end_time)
    else:
        await handler.async_log_success_event(kwargs, response, start_time, end_time)

    cached_value = cache.get_cache(key=f"{model_group}_map")
    latency = cached_value[deployment_id]["latency"][0]
    assert latency == 1.25
    assert isinstance(latency, float)
    json.dumps(cached_value)
