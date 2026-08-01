"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

from typing import List, Optional, cast

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import get_prompt_cache_min_tokens, is_prompt_caching_valid_prompt

from ..prompt_caching_cache import PromptCachingCache


def _get_min_token_count_for_deployments(healthy_deployments: list[dict]) -> int:
    """
    Returns the lowest minimum cacheable prefix across a model group.

    This gate only decides whether the cache lookup is worth doing. It cannot cause a wrong pin,
    because a deployment is only pinned when the cache already holds an entry for the prefix, and
    entries are written by `async_log_success_event` against the deployment's real model. A model
    that will not cache a prefix never records one, so there is nothing to pin it to.

    That makes the lowest minimum in the group the correct threshold rather than the highest.
    `model` here is the model-group alias the operator chose, not a model name, so the threshold
    has to come from the deployments themselves, and a group may mix models whose minimums differ.
    Taking the highest would skip the lookup for a prefix a lower-minimum member genuinely cached,
    losing a cache hit it had earned. The lowest can only cost a lookup that finds nothing.
    """
    return min(
        (
            get_prompt_cache_min_tokens(model=deployment["litellm_params"]["model"])
            for deployment in healthy_deployments
            if deployment.get("litellm_params", {}).get("model")
        ),
        default=DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
    )


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
            min_token_count=_get_min_token_count_for_deployments(healthy_deployments),
        ):
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
        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object", None)

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
