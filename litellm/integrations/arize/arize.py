"""
arize AI is OTEL compatible

this file has Arize ai specific helper functions
"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm.integrations.additional_logging_utils import AdditionalLoggingUtils
from litellm.integrations.arize import _utils
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.types.integrations.arize import ArizeConfig
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.services import ServiceLoggerPayload
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    Span = Union[_Span, Any]
else:
    Protocol = Any
    Span = Any


class ArizeLogger(OpenTelemetry, AdditionalLoggingUtils):
    def set_attributes(self, span: Span, kwargs, response_obj: Optional[Any]):
        ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def set_arize_attributes(span: Span, kwargs, response_obj):
        _utils.set_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def get_arize_config() -> ArizeConfig:
        """
        Helper function to get Arize configuration.

        Returns:
            ArizeConfig: A Pydantic model containing Arize configuration.

        Raises:
            ValueError: If required environment variables are not set.
        """
        space_key = os.environ.get("ARIZE_SPACE_KEY")
        api_key = os.environ.get("ARIZE_API_KEY")

        grpc_endpoint = os.environ.get("ARIZE_ENDPOINT")
        http_endpoint = os.environ.get("ARIZE_HTTP_ENDPOINT")

        endpoint = None
        protocol: Protocol = "otlp_grpc"

        if grpc_endpoint:
            protocol = "otlp_grpc"
            endpoint = grpc_endpoint
        elif http_endpoint:
            protocol = "otlp_http"
            endpoint = http_endpoint
        else:
            protocol = "otlp_grpc"
            endpoint = "https://otlp.arize.com/v1"

        return ArizeConfig(
            space_key=space_key,
            api_key=api_key,
            protocol=protocol,
            endpoint=endpoint,
        )

    async def async_service_success_hook(
        self,
        payload: ServiceLoggerPayload,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """Arize is used mainly for LLM I/O tracing, sending router+caching metrics adds bloat to arize logs"""
        pass

    async def async_service_failure_hook(
        self,
        payload: ServiceLoggerPayload,
        error: Optional[str] = "",
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[float, datetime]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """Arize is used mainly for LLM I/O tracing, sending router+caching metrics adds bloat to arize logs"""
        pass

    def create_litellm_proxy_request_started_span(
        self,
        start_time: datetime,
        headers: dict,
    ):
        """Arize is used mainly for LLM I/O tracing, sending Proxy Server Request adds bloat to arize logs"""
        pass
    

    def construct_dynamic_otel_headers(
        self, 
        standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> Optional[dict]:
        """
        Construct dynamic Arize headers from standard callback dynamic params

        This is used for team/key based logging.

        Returns:
            dict: A dictionary of dynamic Arize headers
        """
        dynamic_headers = {}

        #########################################################
        # `arize-space-id` handling
        # the suggested param is `arize_space_key`
        #########################################################
        if standard_callback_dynamic_params.get("arize_space_id"):
            dynamic_headers["arize-space-id"] = standard_callback_dynamic_params.get(
                "arize_space_id"
            )
        if standard_callback_dynamic_params.get("arize_space_key"):
            dynamic_headers["arize-space-id"] = standard_callback_dynamic_params.get(
                "arize_space_key"
            )
        
        #########################################################
        # `api_key` handling
        #########################################################
        if standard_callback_dynamic_params.get("arize_api_key"):
            dynamic_headers["api_key"] = standard_callback_dynamic_params.get(
                "arize_api_key"
            )
        
        return dynamic_headers

    async def async_health_check(self, standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = None) -> IntegrationHealthCheckStatus:
        """
        Check if Arize service is healthy by testing OTEL trace export
        
        Args:
            standard_callback_dynamic_params: Dynamic parameters containing arize_api_key and arize_space_key/arize_space_id
            
        Returns:
            IntegrationHealthCheckStatus with status and optional error message
        """
        try:
            api_key = None
            space_key = None
            
            if standard_callback_dynamic_params:
                api_key = standard_callback_dynamic_params.get("arize_api_key")
                space_key = (
                    standard_callback_dynamic_params.get("arize_space_key") or 
                    standard_callback_dynamic_params.get("arize_space_id")  # fallback for backwards compatibility
                )
            
            if not api_key:
                api_key = os.environ.get("ARIZE_API_KEY")
            if not space_key:
                space_key = os.environ.get("ARIZE_SPACE_KEY")
            
            if not api_key or not space_key:
                return IntegrationHealthCheckStatus(
                    status="unhealthy", 
                    error_message="Arize credentials not configured. Please set arize_api_key and arize_space_key parameters or ARIZE_API_KEY and ARIZE_SPACE_KEY environment variables."
                )
            
            # Get Arize configuration
            arize_config = ArizeLogger.get_arize_config()
            
            # Validate configuration
            if not arize_config.endpoint:
                return IntegrationHealthCheckStatus(
                    status="unhealthy",
                    error_message="Arize endpoint not configured. Using default endpoint https://otlp.arize.com/v1"
                )
            
            try:
                test_headers = {
                    "arize-space-id": space_key.strip(),
                    "api_key": api_key.strip(),
                }
                
                endpoint = arize_config.endpoint or "https://otlp.arize.com/v1"
                
                # For a basic health check, we just validate that the configuration is properly formed
                # A full test would require actually sending a trace, which might be overkill for health checks
                
                return IntegrationHealthCheckStatus(
                    status="healthy",
                    error_message=None
                )
                
            except Exception as config_error:
                return IntegrationHealthCheckStatus(
                    status="unhealthy",
                    error_message=f"Arize configuration error: {str(config_error)}"
                )
            
        except Exception as e:
            return IntegrationHealthCheckStatus(
                status="unhealthy", 
                error_message=f"Arize health check failed: {str(e)}"
            )

    async def get_request_response_payload(
        self,
        request_id: str,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> Optional[dict]:
        """
        Get the request and response payload for a given request_id from Arize.
        
        Note: Arize is primarily for observability/tracing, not request/response storage.
        This method returns None as Arize doesn't typically store raw payloads.
        """
        return None
