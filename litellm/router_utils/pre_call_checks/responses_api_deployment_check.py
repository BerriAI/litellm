"""
For Responses API, we need routing affinity when a user sends a previous_response_id.

eg. If proxy admins are load balancing between N gpt-4.1-turbo deployments, and a user sends a previous_response_id,
we want to route to the same gpt-4.1-turbo deployment.

This is different from the normal behavior of the router, which does not have routing affinity for previous_response_id.


If previous_response_id is provided, route to the deployment that returned the previous response
"""

from typing import List, Optional

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, StandardLoggingPayload


class ResponsesApiDeploymentCheck(CustomLogger):
    RESPONSES_API_RESPONSE_MODEL_ID_CACHE_KEY = (
        "litellm_responses_api_response_model_id"
    )

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
        request_kwargs = request_kwargs or {}
        previous_response_id = request_kwargs.get("previous_response_id", None)
        if previous_response_id is None:
            return healthy_deployments

        model_id = await self.async_get_response_id_from_cache(
            response_id=previous_response_id,
        )
        if model_id is None:
            return healthy_deployments

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
            call_type != CallTypes.responses.value
            and call_type != CallTypes.aresponses.value
        ):  # only use response id checks for responses api
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.responses_api_deployment_check: skipping adding response_id to cache, CALL TYPE IS NOT RESPONSES"
            )
            return

        response_id = getattr(response_obj, "id", None)
        model_id = standard_logging_object["model_id"]

        if response_id is None or model_id is None:
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.responses_api_deployment_check: skipping adding response_id to cache, RESPONSE ID OR MODEL ID IS NONE"
            )
            return

        await self.async_add_response_id_to_cache(
            response_id=response_id,
            model_id=model_id,
        )

        return

    async def async_add_response_id_to_cache(
        self,
        response_id: str,
        model_id: str,
    ):
        await self.cache.async_set_cache(
            key=self.get_cache_key_for_response_id(response_id),
            value=model_id,
        )

    async def async_get_response_id_from_cache(self, response_id: str) -> Optional[str]:
        cache_value = await self.cache.async_get_cache(
            key=self.get_cache_key_for_response_id(response_id),
        )
        if cache_value is None:
            return None
        return str(cache_value)

    def get_cache_key_for_response_id(self, response_id: str) -> str:
        return f"{self.RESPONSES_API_RESPONSE_MODEL_ID_CACHE_KEY}:{response_id}"
