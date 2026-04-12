"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

from typing import List, Optional, cast

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import is_prompt_caching_valid_prompt

from ..prompt_caching_cache import PromptCachingCache


class PromptCachingDeploymentCheck(CustomLogger):
    def __init__(self, cache: DualCache):
        self.cache = cache

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        if messages is not None and is_prompt_caching_valid_prompt(
            messages=messages,
            model=model,
        ):  # prompt > 1024 tokens
            prompt_cache = PromptCachingCache(
                cache=self.cache,
            )

            model_id_dict = await prompt_cache.async_get_model_id(
                messages=cast(List[AllMessageValues], messages),
                tools=None,
            )
            if model_id_dict is not None:
                model_id = model_id_dict["model_id"]
                for deployment in healthy_deployments:
                    if deployment["model_info"]["id"] == model_id:
                        return [deployment]

        return healthy_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )

        if standard_logging_object is None:
            return

        call_type = standard_logging_object["call_type"]

        if (
            call_type != CallTypes.completion.value
            and call_type != CallTypes.acompletion.value
            and call_type != CallTypes.anthropic_messages.value
        ):  # only use prompt caching for completion calls
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, CALL TYPE IS NOT COMPLETION or ANTHROPIC MESSAGE"
            )
            return

        model = standard_logging_object["model"]
        messages = standard_logging_object["messages"]
        model_id = standard_logging_object["model_id"]

        if messages is None or not isinstance(messages, list):
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MESSAGES IS NOT A LIST"
            )
            return
        if model_id is None:
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MODEL ID IS NONE"
            )
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
