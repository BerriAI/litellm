import base64
import json  # <--- NEW
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.integrations.langfuse.langfuse_otel_attributes import (
    LangfuseLLMObsOTELAttributes,
)
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.types.integrations.langfuse_otel import (
    LangfuseSpanAttributes,
)
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


LANGFUSE_CLOUD_EU_ENDPOINT = "https://cloud.langfuse.com/api/public/otel"
LANGFUSE_CLOUD_US_ENDPOINT = "https://us.cloud.langfuse.com/api/public/otel"


class LangfuseOtelLogger(OpenTelemetry):
    def __init__(self, config=None, *args, **kwargs):
        # Prevent LangfuseOtelLogger from modifying global environment variables by constructing config manually
        # and passing it to the parent OpenTelemetry class
        if config is None:
            config = self._create_open_telemetry_config_from_langfuse_env()
        super().__init__(config=config, *args, **kwargs)

    @staticmethod
    def set_langfuse_otel_attributes(span: Span, kwargs, response_obj):
        """
        Sets OpenTelemetry span attributes for Langfuse observability.
        Uses the same attribute setting logic as Arize Phoenix for consistency.
        """

        _utils.set_attributes(span, kwargs, response_obj, LangfuseLLMObsOTELAttributes)

        #########################################################
        # Set Langfuse specific attributes
        #########################################################
        LangfuseOtelLogger._set_langfuse_specific_attributes(
            span=span, kwargs=kwargs, response_obj=response_obj
        )
        return

    @staticmethod
    def _extract_langfuse_metadata(kwargs: dict) -> dict:
        """
        Extracts Langfuse metadata from the standard LiteLLM kwargs structure.

        1. Reads kwargs["litellm_params"]["metadata"] if present and is a dict.
        2. Enriches it with any `langfuse_*` request-header params via the
           existing LangFuseLogger.add_metadata_from_header helper so that proxy
           users get identical behaviour across vanilla and OTEL integrations.
        """
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata") or {}
        # Ensure we only work with dicts
        if metadata is None or not isinstance(metadata, dict):
            metadata = {}

        # Re-use header extraction logic from the vanilla logger if available
        try:
            from litellm.integrations.langfuse.langfuse import (
                LangFuseLogger as _LFLogger,
            )

            metadata = _LFLogger.add_metadata_from_header(litellm_params, metadata)  # type: ignore
        except Exception:
            # Fallback silently if import fails; header enrichment just won't happen
            pass

        return metadata

    @staticmethod
    def _set_metadata_attributes(span: Span, metadata: dict):
        """Helper to set metadata attributes from mapping."""
        from litellm.integrations.arize._utils import safe_set_attribute

        mapping = {
            "generation_name": LangfuseSpanAttributes.GENERATION_NAME,
            "generation_id": LangfuseSpanAttributes.GENERATION_ID,
            "parent_observation_id": LangfuseSpanAttributes.PARENT_OBSERVATION_ID,
            "version": LangfuseSpanAttributes.GENERATION_VERSION,
            "mask_input": LangfuseSpanAttributes.MASK_INPUT,
            "mask_output": LangfuseSpanAttributes.MASK_OUTPUT,
            "trace_user_id": LangfuseSpanAttributes.TRACE_USER_ID,
            "session_id": LangfuseSpanAttributes.SESSION_ID,
            "tags": LangfuseSpanAttributes.TAGS,
            "trace_name": LangfuseSpanAttributes.TRACE_NAME,
            "trace_id": LangfuseSpanAttributes.TRACE_ID,
            "trace_metadata": LangfuseSpanAttributes.TRACE_METADATA,
            "trace_version": LangfuseSpanAttributes.TRACE_VERSION,
            "trace_release": LangfuseSpanAttributes.TRACE_RELEASE,
            "existing_trace_id": LangfuseSpanAttributes.EXISTING_TRACE_ID,
            "update_trace_keys": LangfuseSpanAttributes.UPDATE_TRACE_KEYS,
            "debug_langfuse": LangfuseSpanAttributes.DEBUG_LANGFUSE,
        }

        for key, enum_attr in mapping.items():
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
                if key == "trace_id" and isinstance(value, str):
                    # trace_id must be 32 hex char no dashes for langfuse : Litellm sends uuid with dashes (might be breaking at some point)
                    value = value.replace("-", "")

                if isinstance(value, (list, dict)):
                    try:
                        value = json.dumps(value)
                    except Exception:
                        value = str(value)
                safe_set_attribute(span, enum_attr.value, value)

    @staticmethod
    def _set_observation_output(span: Span, response_obj):
        """Helper to set observation output attributes."""
        from litellm.integrations.arize._utils import safe_set_attribute
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        if not response_obj or not hasattr(response_obj, "get"):
            return

        choices = response_obj.get("choices", [])
        if choices:
            first_choice = choices[0]
            message = first_choice.get("message", {})
            tool_calls = message.get("tool_calls")
            if tool_calls:
                transformed_tool_calls = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    arguments_str = function.get("arguments", "{}")
                    try:
                        arguments_obj = (
                            json.loads(arguments_str)
                            if isinstance(arguments_str, str)
                            else arguments_str
                        )
                    except json.JSONDecodeError:
                        arguments_obj = {}
                    langfuse_tool_call = {
                        "id": response_obj.get("id", ""),
                        "name": function.get("name", ""),
                        "call_id": tool_call.get("id", ""),
                        "type": "function_call",
                        "arguments": arguments_obj,
                    }
                    transformed_tool_calls.append(langfuse_tool_call)
                safe_set_attribute(
                    span,
                    LangfuseSpanAttributes.OBSERVATION_OUTPUT.value,
                    safe_dumps(transformed_tool_calls),
                )
            else:
                output_data = {}
                if message.get("role"):
                    output_data["role"] = message.get("role")
                if message.get("content") is not None:
                    output_data["content"] = message.get("content")
                if output_data:
                    safe_set_attribute(
                        span,
                        LangfuseSpanAttributes.OBSERVATION_OUTPUT.value,
                        safe_dumps(output_data),
                    )

        output = response_obj.get("output", [])
        if output:
            output_items_data: list[dict] = []
            for item in output:
                if hasattr(item, "type"):
                    item_type = item.type
                    if item_type == "reasoning" and hasattr(item, "summary"):
                        for summary in item.summary:
                            if hasattr(summary, "text"):
                                output_items_data.append(
                                    {
                                        "role": "reasoning_summary",
                                        "content": summary.text,
                                    }
                                )
                    elif item_type == "message":
                        output_items_data.append(
                            {
                                "role": getattr(item, "role", "assistant"),
                                "content": getattr(
                                    getattr(item, "content", [{}])[0], "text", ""
                                ),
                            }
                        )
                    elif item_type == "function_call":
                        arguments_str = getattr(item, "arguments", "{}")
                        arguments_obj = (
                            json.loads(arguments_str)
                            if isinstance(arguments_str, str)
                            else arguments_str
                        )
                        langfuse_tool_call = {
                            "id": getattr(item, "id", ""),
                            "name": getattr(item, "name", ""),
                            "call_id": getattr(item, "call_id", ""),
                            "type": "function_call",
                            "arguments": arguments_obj,
                        }
                        output_items_data.append(langfuse_tool_call)
            if output_items_data:
                safe_set_attribute(
                    span,
                    LangfuseSpanAttributes.OBSERVATION_OUTPUT.value,
                    safe_dumps(output_items_data),
                )

    @staticmethod
    def _set_langfuse_specific_attributes(span: Span, kwargs, response_obj):
        """
        Sets Langfuse specific metadata attributes onto the OTEL span.

        All keys supported by the vanilla Langfuse integration are mapped to
        OTEL-safe attribute names defined in LangfuseSpanAttributes.  Complex
        values (lists/dicts) are serialised to JSON strings for OTEL
        compatibility.
        """
        from litellm.integrations.arize._utils import safe_set_attribute
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        langfuse_environment = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
        if langfuse_environment:
            safe_set_attribute(
                span,
                LangfuseSpanAttributes.LANGFUSE_ENVIRONMENT.value,
                langfuse_environment,
            )

        metadata = LangfuseOtelLogger._extract_langfuse_metadata(kwargs)
        LangfuseOtelLogger._set_metadata_attributes(span=span, metadata=metadata)

        messages = kwargs.get("messages")
        if messages:
            safe_set_attribute(
                span,
                LangfuseSpanAttributes.OBSERVATION_INPUT.value,
                safe_dumps(messages),
            )

        LangfuseOtelLogger._set_observation_output(span=span, response_obj=response_obj)

    @staticmethod
    def _get_langfuse_otel_host() -> Optional[str]:
        """
        Returns the Langfuse OTEL host based on environment variables.

        Returned in the following order of precedence:
        1. LANGFUSE_OTEL_HOST
        2. LANGFUSE_HOST
        """
        return os.environ.get("LANGFUSE_OTEL_HOST") or os.environ.get("LANGFUSE_HOST")

    def _create_open_telemetry_config_from_langfuse_env(self) -> OpenTelemetryConfig:
        """
        Creates OpenTelemetryConfig from Langfuse environment variables.
        Does NOT modify global environment variables.
        """
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", None)
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", None)

        if not public_key or not secret_key:
            # If no keys, return default from env (likely logging to console or something else)
            return OpenTelemetryConfig.from_env()

        # Determine endpoint - default to US cloud
        langfuse_host = LangfuseOtelLogger._get_langfuse_otel_host()

        if langfuse_host:
            # If LANGFUSE_HOST is provided, construct OTEL endpoint from it
            if not langfuse_host.startswith("http"):
                langfuse_host = "https://" + langfuse_host
            endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel"
            verbose_logger.debug(f"Using Langfuse OTEL endpoint from host: {endpoint}")
        else:
            # Default to US cloud endpoint
            endpoint = LANGFUSE_CLOUD_US_ENDPOINT
            verbose_logger.debug(f"Using Langfuse US cloud endpoint: {endpoint}")

        auth_header = LangfuseOtelLogger._get_langfuse_authorization_header(
            public_key=public_key, secret_key=secret_key
        )
        otlp_auth_headers = f"Authorization={auth_header}"

        return OpenTelemetryConfig(
            exporter="otlp_http",
            endpoint=endpoint,
            headers=otlp_auth_headers,
        )

    @staticmethod
    def get_langfuse_otel_config() -> "OpenTelemetryConfig":
        """
        Retrieves the Langfuse OpenTelemetry configuration based on environment variables.

        Environment Variables:
            LANGFUSE_PUBLIC_KEY: Required. Langfuse public key for authentication.
            LANGFUSE_SECRET_KEY: Required. Langfuse secret key for authentication.
            LANGFUSE_HOST: Optional. Custom Langfuse host URL. Defaults to US cloud.

        Returns:
            OpenTelemetryConfig: A Pydantic model containing Langfuse OTEL configuration.

        Raises:
            ValueError: If required keys are missing.
        """
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", None)
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", None)

        if not public_key or not secret_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set for Langfuse OpenTelemetry integration."
            )

        # Determine endpoint - default to US cloud
        langfuse_host = LangfuseOtelLogger._get_langfuse_otel_host()

        if langfuse_host:
            # If LANGFUSE_HOST is provided, construct OTEL endpoint from it
            if not langfuse_host.startswith("http"):
                langfuse_host = "https://" + langfuse_host
            endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel"
            verbose_logger.debug(f"Using Langfuse OTEL endpoint from host: {endpoint}")
        else:
            # Default to US cloud endpoint
            endpoint = LANGFUSE_CLOUD_US_ENDPOINT
            verbose_logger.debug(f"Using Langfuse US cloud endpoint: {endpoint}")

        auth_header = LangfuseOtelLogger._get_langfuse_authorization_header(
            public_key=public_key, secret_key=secret_key
        )
        otlp_auth_headers = f"Authorization={auth_header}"

        # Prevent modification of global env vars which causes leakage
        # os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
        # os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = otlp_auth_headers

        return OpenTelemetryConfig(
            exporter="otlp_http",
            endpoint=endpoint,
            headers=otlp_auth_headers,
        )

    @staticmethod
    def _get_langfuse_authorization_header(public_key: str, secret_key: str) -> str:
        """
        Get the authorization header for Langfuse OpenTelemetry.
        """
        auth_string = f"{public_key}:{secret_key}"
        auth_header = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {auth_header}"

    def construct_dynamic_otel_headers(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> Optional[dict]:
        """
        Construct dynamic Langfuse headers from standard callback dynamic params

        This is used for team/key based logging.

        Returns:
            dict: A dictionary of dynamic Langfuse headers
        """
        dynamic_headers = {}

        dynamic_langfuse_public_key = standard_callback_dynamic_params.get(
            "langfuse_public_key"
        )
        dynamic_langfuse_secret_key = standard_callback_dynamic_params.get(
            "langfuse_secret_key"
        )
        if dynamic_langfuse_public_key and dynamic_langfuse_secret_key:
            auth_header = LangfuseOtelLogger._get_langfuse_authorization_header(
                public_key=dynamic_langfuse_public_key,
                secret_key=dynamic_langfuse_secret_key,
            )
            dynamic_headers["Authorization"] = auth_header

        return dynamic_headers

    def create_litellm_proxy_request_started_span(
        self,
        start_time: datetime,
        headers: dict,
    ) -> Optional[Span]:
        """
        Override to prevent creating empty proxy request spans.

        Langfuse should only receive spans for actual LLM calls, not for
        internal proxy operations (auth, postgres, proxy_pre_call, etc.).

        By returning None, we prevent the parent span from being created,
        which in turn prevents empty traces from being sent to Langfuse.
        """
        return None

    async def async_service_success_hook(self, *args, **kwargs):
        """
        Langfuse should not receive service success logs.
        """
        pass

    async def async_service_failure_hook(self, *args, **kwargs):
        """
        Langfuse should not receive service failure logs.
        """
        pass
