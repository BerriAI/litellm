from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry.trace import Status, StatusCode
from typing_extensions import override

from litellm._logging import verbose_logger
from litellm.integrations._types.open_inference import SpanAttributes as OpenInferenceSpanAttributes
from litellm.integrations.arize import _utils
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.integrations.weave_otel import WeaveOtelConfig, WeaveSpanAttributes
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span


# Weave OTEL endpoint
# Multi-tenant cloud: https://trace.wandb.ai/otel/v1/traces
# Dedicated cloud: https://<your-subdomain>.wandb.io/traces/otel/v1/traces
WEAVE_BASE_URL = "https://trace.wandb.ai"
WEAVE_OTEL_ENDPOINT = "/otel/v1/traces"


class WeaveLLMObsOTELAttributes(BaseLLMObsOTELAttributes):
    """
    Weave-specific LLM observability OTEL attributes.

    Weave automatically maps attributes from multiple frameworks including
    GenAI, OpenInference, Langfuse, and others.
    """

    @staticmethod
    @override
    def set_messages(span: "Span", kwargs: dict[str, Any]):
        """Set input messages as span attributes using OpenInference conventions."""

        messages = kwargs.get("messages") or []
        optional_params = kwargs.get("optional_params") or {}

        prompt = {"messages": messages}
        functions = optional_params.get("functions")
        tools = optional_params.get("tools")
        if functions is not None:
            prompt["functions"] = functions
        if tools is not None:
            prompt["tools"] = tools
        safe_set_attribute(span, OpenInferenceSpanAttributes.INPUT_VALUE, json.dumps(prompt))


def _set_weave_specific_attributes(span: Span, kwargs: dict[str, Any], response_obj: Any):
    """
    Sets Weave-specific metadata attributes onto the OTEL span.

    Based on Weave's OTEL attribute mappings from:
    https://github.com/wandb/weave/blob/master/weave/trace_server/opentelemetry/constants.py
    """

    # Extract all needed data upfront
    litellm_params = kwargs.get("litellm_params") or {}
    # optional_params = kwargs.get("optional_params") or {}
    metadata = kwargs.get("metadata") or {}
    model = kwargs.get("model") or ""
    custom_llm_provider = litellm_params.get("custom_llm_provider") or ""

    # Weave supports a custom display name and will default to the model name if not provided.
    display_name = metadata.get("display_name")
    if not display_name and model:
        if custom_llm_provider:
            display_name = f"{custom_llm_provider}/{model}"
        else:
            display_name = model
    if display_name:
        display_name = display_name.replace("/", "__")
        safe_set_attribute(span, WeaveSpanAttributes.DISPLAY_NAME.value, display_name)

    # Weave threads are OpenInference sessions.
    if (session_id := metadata.get("session_id")) is not None:
        if isinstance(session_id, (list, dict)):
            session_id = safe_dumps(session_id)
        safe_set_attribute(span, WeaveSpanAttributes.THREAD_ID.value, session_id)
        safe_set_attribute(span, WeaveSpanAttributes.IS_TURN.value, True)

    # Response attributes are already set by _utils.set_attributes,
    # but we override them here to better match Weave's expectations
    if response_obj:
        output_dict = None
        if hasattr(response_obj, "model_dump"):
            output_dict = response_obj.model_dump()
        elif hasattr(response_obj, "get"):
            output_dict = response_obj

        if output_dict:
            safe_set_attribute(span, OpenInferenceSpanAttributes.OUTPUT_VALUE, safe_dumps(output_dict))


def _get_weave_authorization_header(api_key: str) -> str:
    """
    Get the authorization header for Weave OpenTelemetry.

    Weave uses Basic auth with format: api:<WANDB_API_KEY>
    """
    auth_string = f"api:{api_key}"
    auth_header = base64.b64encode(auth_string.encode()).decode()
    return f"Basic {auth_header}"


def get_weave_otel_config() -> WeaveOtelConfig:
    """
    Retrieves the Weave OpenTelemetry configuration based on environment variables.

    Environment Variables:
        WANDB_API_KEY: Required. W&B API key for authentication.
        WANDB_PROJECT_ID: Required. Project ID in format <entity>/<project_name>.
        WANDB_HOST: Optional. Custom Weave host URL. Defaults to cloud endpoint.

    Returns:
        WeaveOtelConfig: A Pydantic model containing Weave OTEL configuration.

    Raises:
        ValueError: If required environment variables are missing.
    """
    api_key = os.getenv("WANDB_API_KEY")
    project_id = os.getenv("WANDB_PROJECT_ID")
    host = os.getenv("WANDB_HOST")

    if not api_key:
        raise ValueError("WANDB_API_KEY must be set for Weave OpenTelemetry integration.")

    if not project_id:
        raise ValueError(
            "WANDB_PROJECT_ID must be set for Weave OpenTelemetry integration. Format: <entity>/<project_name>"
        )

    if host:
        if not host.startswith("http"):
            host = "https://" + host
        # Self-managed instances use a different path
        endpoint = host.rstrip("/") + WEAVE_OTEL_ENDPOINT
        verbose_logger.debug(f"Using Weave OTEL endpoint from host: {endpoint}")
    else:
        endpoint = WEAVE_BASE_URL + WEAVE_OTEL_ENDPOINT
        verbose_logger.debug(f"Using Weave cloud endpoint: {endpoint}")

    # Weave uses Basic auth with format: api:<WANDB_API_KEY>
    auth_header = _get_weave_authorization_header(api_key=api_key)
    otlp_auth_headers = f"Authorization={auth_header},project_id={project_id}"

    # Set standard OTEL environment variables
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = otlp_auth_headers

    return WeaveOtelConfig(
        otlp_auth_headers=otlp_auth_headers,
        endpoint=endpoint,
        project_id=project_id,
        protocol="otlp_http",
    )


def set_weave_otel_attributes(span: Span, kwargs: dict[str, Any], response_obj: Any):
    """
    Sets OpenTelemetry span attributes for Weave observability.
    Uses the same attribute setting logic as other OTEL integrations for consistency.
    """
    _utils.set_attributes(span, kwargs, response_obj, WeaveLLMObsOTELAttributes)
    _set_weave_specific_attributes(span=span, kwargs=kwargs, response_obj=response_obj)


class WeaveOtelLogger(OpenTelemetry):
    """
    Weave (W&B) OpenTelemetry Logger for LiteLLM.

    Sends LLM traces to Weave via the OpenTelemetry Protocol (OTLP).

    Environment Variables:
        WANDB_API_KEY: Required. Weights & Biases API key for authentication.
        WANDB_PROJECT_ID: Required. Project ID in format <entity>/<project_name>.
        WANDB_HOST: Optional. Custom Weave host URL. Defaults to cloud endpoint.

    Usage:
        litellm.callbacks = ["weave_otel"]

        Or manually:
        from litellm.integrations.weave.weave_otel import WeaveOtelLogger
        weave_logger = WeaveOtelLogger(callback_name="weave_otel")
        litellm.callbacks = [weave_logger]

    Reference:
        https://docs.wandb.ai/weave/guides/tracking/otel
    """

    def __init__(
        self,
        config: Optional[OpenTelemetryConfig] = None,
        callback_name: Optional[str] = "weave_otel",
        **kwargs,
    ):
        """
        Initialize WeaveOtelLogger.

        If config is not provided, automatically configures from environment variables
        (WANDB_API_KEY, WANDB_PROJECT_ID, WANDB_HOST) via get_weave_otel_config().
        """
        if config is None:
            # Auto-configure from Weave environment variables
            weave_config = get_weave_otel_config()

            config = OpenTelemetryConfig(
                exporter=weave_config.protocol,
                endpoint=weave_config.endpoint,
                headers=weave_config.otlp_auth_headers,
            )

        super().__init__(config=config, callback_name=callback_name, **kwargs)

    def _maybe_log_raw_request(self, kwargs, response_obj, start_time, end_time, parent_span):
        """
        Override to skip creating the raw_gen_ai_request child span.

        For Weave, we only want a single span per LLM call. The parent span
        already contains all the necessary attributes, so the child span
        is redundant.
        """
        pass

    def _start_primary_span(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
        context,
        parent_span=None,
    ):
        """
        Override to always create a child span instead of reusing the parent span.

        This ensures that wrapper spans (like "B", "C", "D", "E") remain separate
        from the LiteLLM LLM call spans, creating proper nesting in Weave.
        """

        otel_tracer = self.get_tracer_to_use_for_request(kwargs)
        # Always create a new child span, even if parent_span is provided
        # This ensures wrapper spans remain separate from LLM call spans
        span = otel_tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=context,
        )
        span.set_status(Status(StatusCode.OK))
        self.set_attributes(span, kwargs, response_obj)
        span.end(end_time=self._to_ns(end_time))
        return span

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        """
        Override to prevent ending externally created parent spans.

        When wrapper spans (like "B", "C", "D", "E") are provided as parent spans,
        they should be managed by the user code, not ended by LiteLLM.
        """

        verbose_logger.debug(
            "Weave OpenTelemetry Logger: Logging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )
        ctx, parent_span = self._get_span_context(kwargs)

        # Always create a child span (handled by _start_primary_span override)
        primary_span_parent = None

        # 1. Primary span
        span = self._start_primary_span(kwargs, response_obj, start_time, end_time, ctx, primary_span_parent)

        # 2. Raw-request sub-span (skipped for Weave via _maybe_log_raw_request override)
        self._maybe_log_raw_request(kwargs, response_obj, start_time, end_time, span)

        # 3. Guardrail span
        self._create_guardrail_span(kwargs=kwargs, context=ctx)

        # 4. Metrics & cost recording
        self._record_metrics(kwargs, response_obj, start_time, end_time)

        # 5. Semantic logs.
        if self.config.enable_events:
            self._emit_semantic_logs(kwargs, response_obj, span)

        # 6. Don't end parent span - it's managed by user code
        # Since we always create a child span (never reuse parent), the parent span
        # lifecycle is owned by the user. This prevents double-ending of wrapper spans
        # like "B", "C", "D", "E" that users create and manage themselves.

    def construct_dynamic_otel_headers(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> dict | None:
        """
        Construct dynamic Weave headers from standard callback dynamic params.

        This is used for team/key based logging.

        Returns:
            dict: A dictionary of dynamic Weave headers
        """
        dynamic_headers = {}

        dynamic_wandb_api_key = standard_callback_dynamic_params.get("wandb_api_key")
        dynamic_weave_project_id = standard_callback_dynamic_params.get("weave_project_id")

        if dynamic_wandb_api_key:
            auth_header = _get_weave_authorization_header(
                api_key=dynamic_wandb_api_key,
            )
            dynamic_headers["Authorization"] = auth_header

        if dynamic_weave_project_id:
            dynamic_headers["project_id"] = dynamic_weave_project_id

        return dynamic_headers if dynamic_headers else None
