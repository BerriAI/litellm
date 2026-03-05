"""
Cursor Cloud Agents API - Pass-through Logging Handler

Transforms Cursor API responses into standardized logging payloads
so they appear cleanly in the LiteLLM Logs page.
"""

from datetime import datetime
from typing import Dict

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import StandardPassThroughResponseObject


CURSOR_AGENT_ENDPOINTS: Dict[str, str] = {
    "POST /v0/agents": "cursor:agent:create",
    "GET /v0/agents": "cursor:agent:list",
    "POST /v0/agents/{id}/followup": "cursor:agent:followup",
    "POST /v0/agents/{id}/stop": "cursor:agent:stop",
    "DELETE /v0/agents/{id}": "cursor:agent:delete",
    "GET /v0/agents/{id}/conversation": "cursor:agent:conversation",
    "GET /v0/agents/{id}": "cursor:agent:status",
    "GET /v0/me": "cursor:account:info",
    "GET /v0/models": "cursor:models:list",
    "GET /v0/repositories": "cursor:repositories:list",
}


def _classify_cursor_request(method: str, path: str) -> str:
    """Classify a Cursor API request into a readable operation name."""
    normalized = path.rstrip("/")

    for pattern, operation in CURSOR_AGENT_ENDPOINTS.items():
        pat_method, pat_path = pattern.split(" ", 1)
        if method.upper() != pat_method:
            continue

        pat_parts = pat_path.strip("/").split("/")
        req_parts = normalized.strip("/").split("/")

        if len(pat_parts) != len(req_parts):
            continue

        match = True
        for pp, rp in zip(pat_parts, req_parts):
            if pp.startswith("{") and pp.endswith("}"):
                continue
            if pp != rp:
                match = False
                break
        if match:
            return operation

    return f"cursor:{method.lower()}:{normalized}"


class CursorPassthroughLoggingHandler:
    """Handles logging for Cursor Cloud Agents pass-through requests."""

    @staticmethod
    def cursor_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Transform a Cursor API response into a standard logging payload.
        """
        try:
            method = httpx_response.request.method
            path = httpx.URL(url_route).path
            operation = _classify_cursor_request(method, path)

            agent_id = response_body.get("id", "")
            agent_name = response_body.get("name", "")
            agent_status = response_body.get("status", "")

            model_name = f"cursor/{operation}"

            summary_parts = []
            if agent_id:
                summary_parts.append(f"id={agent_id}")
            if agent_name:
                summary_parts.append(f"name={agent_name}")
            if agent_status:
                summary_parts.append(f"status={agent_status}")

            response_summary = ", ".join(summary_parts) if summary_parts else result

            kwargs["model"] = model_name
            kwargs["response_cost"] = 0.0
            logging_obj.model_call_details["model"] = model_name
            logging_obj.model_call_details["custom_llm_provider"] = "cursor"
            logging_obj.model_call_details["response_cost"] = 0.0

            standard_logging_object = get_standard_logging_object_payload(
                kwargs=kwargs,
                init_response_obj=StandardPassThroughResponseObject(
                    response=response_summary
                ),
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                status="success",
            )
            kwargs["standard_logging_object"] = standard_logging_object

            verbose_proxy_logger.debug(
                "Cursor passthrough logging: operation=%s, agent_id=%s",
                operation,
                agent_id,
            )

            return {
                "result": StandardPassThroughResponseObject(
                    response=response_summary
                ),
                "kwargs": kwargs,
            }
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error in Cursor passthrough logging handler: %s", e
            )
            return {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": kwargs,
            }
