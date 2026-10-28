"""
For Responses API, we need routing affinity when a user sends a previous_response_id.

eg. If proxy admins are load balancing between N gpt-4.1-turbo deployments, and a user sends a previous_response_id,
we want to route to the same gpt-4.1-turbo deployment.

This is different from the normal behavior of the router, which does not have routing affinity for previous_response_id.


If previous_response_id is provided, route to the deployment that returned the previous response
"""

import warnings
from collections.abc import Mapping
from typing import Final

from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.common_utils import deployment_ids_failed_in_request
from litellm.types.llms.openai import AllMessageValues


class ResponsesApiDeploymentCheck(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            (
                "ResponsesApiDeploymentCheck is deprecated. "
                "Use DeploymentAffinityCheck(enable_responses_api_affinity=True) instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: Mapping[str, object] | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        request: Final = request_kwargs or {}
        model_id: Final = ResponsesAPIRequestUtils.pinned_response_model_id(
            request, deployment_ids_failed_in_request(request)
        )
        if model_id is None:
            return healthy_deployments

        for deployment in healthy_deployments:
            if deployment["model_info"]["id"] == model_id:
                return [deployment]

        return healthy_deployments
