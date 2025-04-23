"""
For Responses API, we need routing affinity when a user sends a previous_response_id.

eg. If proxy admins are load balancing between N gpt-4.1-turbo deployments, and a user sends a previous_response_id,
we want to route to the same gpt-4.1-turbo deployment.

This is different from the normal behavior of the router, which does not have routing affinity for previous_response_id.


If previous_response_id is provided, route to the deployment that returned the previous response
"""

from typing import List, Optional

from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import AllMessageValues


class ResponsesApiDeploymentCheck(CustomLogger):
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

        decoded_response = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            response_id=previous_response_id,
        )
        model_id = decoded_response.get("model_id")
        if model_id is None:
            return healthy_deployments

        for deployment in healthy_deployments:
            if deployment["model_info"]["id"] == model_id:
                return [deployment]

        return healthy_deployments
