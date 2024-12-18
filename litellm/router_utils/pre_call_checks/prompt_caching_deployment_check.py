"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

from typing import List, Optional, cast

from litellm.cache.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import is_prompt_caching_valid_prompt

from ..prompt_caching_cache import PromptCachingCache


class PromptCachingDeploymentCheck(CustomLogger):
    def __init__(self, cache: DualCache):
        self.cache = cache

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )

        if standard_logging_object is None:
            return

        call_type = standard_logging_object["call_type"]
        if (
            call_type != CallTypes.completion.value
            or call_type != CallTypes.acompletion.value
        ):  # only use prompt caching for completion calls
            return

        model = standard_logging_object["model"]
        messages = standard_logging_object["messages"]
        model_id = standard_logging_object["model_id"]

        if messages is None or not isinstance(messages, list):
            return
        if model_id is None:
            return

        ## PROMPT CACHING - cache model id, if prompt caching valid prompt + provider
        if is_prompt_caching_valid_prompt(
            model=model,
            messages=cast(List[AllMessageValues], messages),
        ):
            cache = PromptCachingCache(
                cache=self.cache,
            )
            await cache.async_add_model_id(
                model_id=model_id,
                messages=messages,
                tools=None,  # [TODO]: add tools once standard_logging_object supports it
            )

        return
