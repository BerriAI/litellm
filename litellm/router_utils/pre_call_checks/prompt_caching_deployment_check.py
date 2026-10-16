"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

from collections.abc import Callable, Mapping
from typing import Final, TypedDict, cast

from typing_extensions import ReadOnly

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.integrations.anthropic_cache_control_hook import (
    AllToolParamValues,
    AnthropicCacheControlHook,
)
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.llms.bedrock.common_utils import bedrock_cache_point_ttl_supported, bedrock_capability_model
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import get_prompt_cache_min_tokens, is_prompt_caching_valid_prompt

from ..prompt_caching_cache import PromptCachingCache


class _PromptCachingSuccessEventKwargs(TypedDict, total=False):
    standard_logging_object: ReadOnly[StandardLoggingPayload | None]
    litellm_params: ReadOnly[Mapping[str, object] | None]


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


def _honors_one_hour_cache_ttl(
    standard_logging_object: StandardLoggingPayload, litellm_params: Mapping[str, object]
) -> bool:
    if standard_logging_object["custom_llm_provider"] == "bedrock":
        configured_model: Final = litellm_params.get("model")
        deployment_model: Final = (
            configured_model if isinstance(configured_model, str) else standard_logging_object["model"]
        )
        capability_model: Final = bedrock_capability_model(deployment_model, litellm_params.get("base_model"))
        return bedrock_cache_point_ttl_supported(capability_model, deployment_model)
    model_map_value: Final = standard_logging_object["model_map_information"]["model_map_value"]
    return model_map_value is not None and (model_map_value.get("cache_creation_input_token_cost_above_1hr") or 0) > 0


class PromptCachingDeploymentCheck(CustomLogger):
    def __init__(self, cache: DualCache, is_priority_group: Callable[[str], bool] | None = None):
        super().__init__()
        self.cache = cache
        self.is_priority_group = is_priority_group

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        if self.is_priority_group is not None and self.is_priority_group(model):
            return healthy_deployments
        if request_kwargs is not None and request_kwargs.get("_target_order") is not None:
            return healthy_deployments
        if not healthy_deployments[1:]:
            return healthy_deployments

        if messages is not None and await offload_token_count(is_prompt_caching_valid_prompt)(
            messages=messages,
            model=model,
            min_token_count=_get_min_token_count_for_deployments(healthy_deployments),
        ):
            prompt_cache: Final = PromptCachingCache(
                cache=self.cache,
            )

            ## AUTO PROMPT CACHING - the breakpoints this request will carry are injected inside
            ## `litellm.acompletion`, after a deployment has been picked, so the affinity key has to
            ## be derived from the messages as they will be sent, not as they arrive here.
            affinity_messages: Final = AnthropicCacheControlHook.messages_with_default_injections(
                messages=cast(list[AllMessageValues], messages),
                models=(
                    deployment["litellm_params"]["model"]
                    for deployment in healthy_deployments
                    if isinstance(deployment.get("litellm_params"), dict) and deployment["litellm_params"].get("model")
                ),
                tools=(
                    cast(  # cast-ok: request_kwargs is untyped; the stand-down scan duck-types every tool it reads
                        list[AllToolParamValues] | None, request_kwargs.get("tools")
                    )
                    if request_kwargs is not None
                    else None
                ),
                enable_prompt_caching=(
                    request_kwargs.get("enable_prompt_caching") is True if request_kwargs is not None else None
                ),
                request_kwargs=request_kwargs,
            )

            model_id_dict: Final = await prompt_cache.async_get_model_id(
                messages=affinity_messages,
                tools=None,
            )
            if model_id_dict is not None:
                model_id: Final = model_id_dict["model_id"]
                for deployment in healthy_deployments:
                    if deployment["model_info"]["id"] == model_id:
                        return [deployment]

        return healthy_deployments

    async def async_log_success_event(
        self,
        kwargs: _PromptCachingSuccessEventKwargs,
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)

        if standard_logging_object is None:
            return

        call_type: Final = standard_logging_object["call_type"]

        if (
            call_type != CallTypes.completion.value
            and call_type != CallTypes.acompletion.value
            and call_type != CallTypes.anthropic_messages.value
        ):  # only use prompt caching for completion calls
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, CALL TYPE IS NOT COMPLETION or ANTHROPIC MESSAGE"
            )
            return

        model: Final = standard_logging_object["model"]
        messages: Final = standard_logging_object["messages"]
        model_id: Final = standard_logging_object["model_id"]

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

        prompt_messages: Final = cast(list[AllMessageValues], messages)
        ## PROMPT CACHING - cache model id, if prompt caching valid prompt + provider
        if await offload_token_count(is_prompt_caching_valid_prompt)(
            model=model,
            messages=prompt_messages,
        ):
            cache: Final = PromptCachingCache(
                cache=self.cache,
            )
            raw_litellm_params: Final = kwargs.get("litellm_params")
            litellm_params: Final[Mapping[str, object]] = (
                raw_litellm_params if isinstance(raw_litellm_params, Mapping) else {}
            )
            await cache.async_add_model_id(
                model_id=model_id,
                messages=prompt_messages,
                tools=None,  # [TODO]: add tools once standard_logging_object supports it
                honors_one_hour_ttl=_honors_one_hour_cache_ttl(standard_logging_object, litellm_params),
            )

        return
