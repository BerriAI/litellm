import json
import os
from collections.abc import AsyncGenerator
from itertools import count
from typing import Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponseStream


class FallbackCostRecorder(CustomLogger):
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[ModelResponseStream, None],
        request_data: dict[str, object],
    ) -> AsyncGenerator[ModelResponseStream, None]:
        log_path: Final = os.environ["LITELLM_FALLBACK_COST_LOG"]
        chunk_indices: Final = count()
        async for item in response:
            hidden_params: Final = getattr(item, "_hidden_params", None)
            usage: Final = getattr(item, "usage", None)
            event_type: Final = getattr(item, "type", None)
            record: Final = {
                "request_model": request_data.get("model"),
                "item_type": type(item).__name__,
                "worker_pid": os.getpid(),
                "response_cost": hidden_params.get("response_cost") if isinstance(hidden_params, dict) else None,
                "usage_cost": getattr(
                    usage,
                    "cost",
                    getattr(getattr(getattr(item, "response", None), "usage", None), "cost", None),
                ),
                "event_type": event_type,
                "chunk_index": next(chunk_indices) if event_type is None else None,
                "has_hidden_params": hasattr(item, "_hidden_params"),
            }
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps(record, default=str) + "\n")
            yield item


proxy_handler_instance = FallbackCostRecorder()
