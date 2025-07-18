# What is this?
## On Success events log cost to CloudZero Unit Cost Analytics - https://github.com/BerriAI/litellm/issues/5773

import json
import os
from typing import Optional

import httpx

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.utils import get_end_user_id_for_cost_tracking, get_utc_datetime


class CloudZeroLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.validate_environment()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_http_handler = HTTPHandler()

    def validate_environment(self):
        """
        Expects
        CLOUDZERO_API_KEY,
        CLOUDZERO_METRIC_NAME,

        Optional:
        CLOUDZERO_API_BASE (defaults to https://api.cloudzero.com)

        in the environment
        """
        missing_keys = []
        if os.getenv("CLOUDZERO_API_KEY", None) is None:
            missing_keys.append("CLOUDZERO_API_KEY")

        if os.getenv("CLOUDZERO_METRIC_NAME", None) is None:
            missing_keys.append("CLOUDZERO_METRIC_NAME")

        if len(missing_keys) > 0:
            raise Exception("Missing keys={} in environment.".format(missing_keys))

    def _get_api_url(self) -> str:
        """Get the CloudZero API URL for posting telemetry data"""
        base_url = os.getenv("CLOUDZERO_API_BASE", "https://api.cloudzero.com")
        metric_name = os.getenv("CLOUDZERO_METRIC_NAME")
        
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        
        return f"{base_url}/unit-cost/v1/telemetry/metric/{metric_name}"

    def _common_logic(self, kwargs: dict, response_obj) -> dict:
        """Extract common data for CloudZero telemetry"""
        timestamp = get_utc_datetime().isoformat()
        cost = kwargs.get("response_cost", 0.0)
        
        # CloudZero expects cost in dollars
        if cost is None:
            cost = 0.0
        
        # Create the telemetry record
        record = {
            "value": cost,
            "timestamp": timestamp
        }
        
        # Optional: Add dimensions/filters for cost attribution
        filters = {}
        
        # Get model information
        model = kwargs.get("model")
        if model:
            filters["model"] = model
        
        # Get user/team information from litellm_params
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata", {}) or {}
        
        # Add user information if available
        user_id = metadata.get("user_api_key_user_id")
        if user_id:
            filters["user_id"] = user_id
        
        # Add team information if available
        team_id = metadata.get("user_api_key_team_id")
        if team_id:
            filters["team_id"] = team_id
        
        # Add organization information if available
        org_id = metadata.get("user_api_key_org_id")
        if org_id:
            filters["org_id"] = org_id
        
        # Add end user information if available
        end_user_id = get_end_user_id_for_cost_tracking(litellm_params)
        if end_user_id:
            filters["end_user_id"] = end_user_id
        
        # Add filters to record if any exist
        if filters:
            record["filters"] = filters
        
        # CloudZero expects records in an array
        returned_val = {
            "records": [record]
        }
        
        verbose_logger.debug(
            "\033[91mLogged CloudZero Object:\n{}\033[0m\n".format(returned_val)
        )
        return returned_val

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Synchronous logging of success events to CloudZero"""
        url = self._get_api_url()
        api_key = os.getenv("CLOUDZERO_API_KEY")
        
        data = self._common_logic(kwargs=kwargs, response_obj=response_obj)
        headers = {
            "Content-Type": "application/json",
            "Authorization": api_key,
        }
        
        try:
            response = self.sync_http_handler.post(
                url=url,
                data=json.dumps(data),
                headers=headers,
            )
            
            response.raise_for_status()
            verbose_logger.debug(f"CloudZero telemetry posted successfully: {response.text}")
        except Exception as e:
            error_response = getattr(e, "response", None)
            if error_response is not None and hasattr(error_response, "text"):
                verbose_logger.debug(f"\nCloudZero Error Message: {error_response.text}")
            # Log the error but don't raise - we don't want to fail the LLM call
            verbose_logger.error(f"Failed to post telemetry to CloudZero: {str(e)}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Asynchronous logging of success events to CloudZero"""
        try:
            verbose_logger.debug("ENTERS CLOUDZERO CALLBACK")
            url = self._get_api_url()
            api_key = os.getenv("CLOUDZERO_API_KEY")
            
            data = self._common_logic(kwargs=kwargs, response_obj=response_obj)
            headers = {
                "Content-Type": "application/json",
                "Authorization": api_key,
            }
        except Exception as e:
            verbose_logger.error(f"Failed to prepare CloudZero telemetry: {str(e)}")
            return
        
        response: Optional[httpx.Response] = None
        try:
            response = await self.async_http_handler.post(
                url=url,
                data=json.dumps(data),
                headers=headers,
            )
            
            response.raise_for_status()
            verbose_logger.debug(f"CloudZero telemetry posted successfully: {response.text}")
        except Exception as e:
            if response is not None and hasattr(response, "text"):
                verbose_logger.debug(f"\nCloudZero Error Message: {response.text}")
            # Log the error but don't raise - we don't want to fail the LLM call
            verbose_logger.error(f"Failed to post telemetry to CloudZero: {str(e)}")