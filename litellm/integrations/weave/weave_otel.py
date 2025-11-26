from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any, Union

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.integrations.weave.weave_otel_attributes import WeaveLLMObsOTELAttributes
from litellm.types.integrations.weave import WeaveOtelConfig, WeaveSpanAttributes
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.integrations.opentelemetry import (
        OpenTelemetryConfig as _OpenTelemetryConfig,
    )
    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = Union[_Span, Any]
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any


# Weave OTEL endpoint
# Multi-tenant cloud: https://trace.wandb.ai/otel/v1/traces
# Self-managed: https://<your-subdomain>.wandb.io/traces/otel/v1/traces
WEAVE_CLOUD_ENDPOINT = "https://trace.wandb.ai/otel/v1/traces"


class WeaveOtelLogger(OpenTelemetry):
    """
    Weave (W&B) OpenTelemetry Logger for LiteLLM.

    Sends LLM traces to Weave via the OpenTelemetry Protocol (OTLP).

    Environment Variables:
        WANDB_API_KEY: Required. Weights & Biases API key for authentication.
        WEAVE_PROJECT_ID: Required. Project ID in format <entity>/<project_name>.
        WEAVE_HOST: Optional. Custom Weave host URL. Defaults to cloud endpoint.

    Usage:
        litellm.callbacks = ["weave_otel"]

    Reference:
        https://docs.wandb.ai/weave/guides/tracking/otel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _maybe_log_raw_request(
        self, kwargs, response_obj, start_time, end_time, parent_span
    ):
        """
        Override to skip creating the raw_gen_ai_request child span.

        For Weave, we only want a single span per LLM call. The parent span
        already contains all the necessary attributes, so the child span
        is redundant.
        """
        pass

    @staticmethod
    def set_weave_otel_attributes(span: Span, kwargs, response_obj):
        """
        Sets OpenTelemetry span attributes for Weave observability.
        Uses the same attribute setting logic as other OTEL integrations for consistency.
        """
        _utils.set_attributes(span, kwargs, response_obj, WeaveLLMObsOTELAttributes)

        # Set Weave-specific attributes
        WeaveOtelLogger._set_weave_specific_attributes(
            span=span, kwargs=kwargs, response_obj=response_obj
        )

    @staticmethod
    def _extract_weave_metadata(kwargs: dict) -> dict:
        """
        Extracts Weave metadata from the standard LiteLLM kwargs structure.

        Reads kwargs["litellm_params"]["metadata"] if present and is a dict.
        """
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata

    @staticmethod
    def _set_metadata_attributes(span: Span, metadata: dict):
        """Helper to set metadata attributes from mapping."""
        from litellm.integrations.arize._utils import safe_set_attribute
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        # Weave-specific attributes
        weave_mapping = {
            "thread_id": WeaveSpanAttributes.THREAD_ID,
            "is_turn": WeaveSpanAttributes.IS_TURN,
            "display_name": WeaveSpanAttributes.DISPLAY_NAME,
        }

        # Standard attributes recognized by Weave
        standard_mapping = {
            "trace_user_id": WeaveSpanAttributes.TRACE_USER_ID,
            "user_id": WeaveSpanAttributes.TRACE_USER_ID,
            "session_id": WeaveSpanAttributes.SESSION_ID,
        }

        for key, enum_attr in {**weave_mapping, **standard_mapping}.items():
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
                if isinstance(value, (list, dict)):
                    value = safe_dumps(value)
                safe_set_attribute(span, enum_attr.value, value)

        # Set general metadata as JSON
        if metadata:
            safe_set_attribute(span, WeaveSpanAttributes.METADATA.value, safe_dumps(metadata))

    @staticmethod
    def _set_token_usage(span: Span, response_obj):
        """Helper to set token usage attributes using OpenInference conventions."""
        from litellm.integrations.arize._utils import safe_set_attribute

        if not response_obj or not hasattr(response_obj, "get"):
            return

        usage = response_obj.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")

            if prompt_tokens is not None:
                safe_set_attribute(span, WeaveSpanAttributes.LLM_TOKEN_COUNT_PROMPT.value, prompt_tokens)
            if completion_tokens is not None:
                safe_set_attribute(span, WeaveSpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value, completion_tokens)
            if total_tokens is not None:
                safe_set_attribute(span, WeaveSpanAttributes.LLM_TOKEN_COUNT_TOTAL.value, total_tokens)

    @staticmethod
    def _set_weave_specific_attributes(span: Span, kwargs, response_obj):
        """
        Sets Weave-specific metadata attributes onto the OTEL span.

        Based on Weave's OTEL attribute mappings from:
        https://github.com/wandb/weave/blob/master/weave/trace_server/opentelemetry/constants.py

        Weave maps these attributes to its internal model:
        - "model": gen_ai.response.model, llm.model_name, ai.model.id
        - "provider": llm.provider, ai.model.provider
        - "kind": openinference.span.kind, weave.span.kind, traceloop.span.kind
        - "model_parameters": gen_ai.request, llm.invocation_parameters
        - "inputs": input.value
        - "outputs": output.value
        """
        from litellm.integrations.arize._utils import safe_set_attribute
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        # Set metadata attributes (thread_id, session_id, etc.)
        metadata = WeaveOtelLogger._extract_weave_metadata(kwargs)
        WeaveOtelLogger._set_metadata_attributes(span=span, metadata=metadata)

        # Get model and provider info
        litellm_params = kwargs.get("litellm_params", {}) or {}
        model = kwargs.get("model", "")
        custom_llm_provider = litellm_params.get("custom_llm_provider", "")
        optional_params = kwargs.get("optional_params", {})

        # === Weave "model" attribute ===
        # Maps from: gen_ai.response.model, llm.model_name, ai.model.id
        if model:
            safe_set_attribute(span, "llm.model_name", model)
            safe_set_attribute(span, "gen_ai.response.model", model)

        # === Weave "provider" attribute ===
        # Maps from: llm.provider, ai.model.provider
        if custom_llm_provider:
            safe_set_attribute(span, "llm.provider", custom_llm_provider)

        # === Weave "kind" attribute ===
        # Maps from: openinference.span.kind, weave.span.kind, traceloop.span.kind
        safe_set_attribute(span, "openinference.span.kind", "LLM")

        # === Weave "model_parameters" attribute ===
        # Maps from: gen_ai.request, llm.invocation_parameters
        if optional_params:
            # Filter out sensitive fields
            params_to_log = {k: v for k, v in optional_params.items() if k != "secret_fields"}
            safe_set_attribute(span, "llm.invocation_parameters", safe_dumps(params_to_log))

        # === Weave "outputs" attribute ===
        # Maps from: output.value - should be the full response object, not just content
        # This overrides what _utils.set_attributes set (which was just message content)
        if response_obj:
            if hasattr(response_obj, "model_dump"):
                # Pydantic model - serialize to dict then JSON
                safe_set_attribute(span, "output.value", safe_dumps(response_obj.model_dump()))
            elif hasattr(response_obj, "get"):
                # Dict-like object
                safe_set_attribute(span, "output.value", safe_dumps(response_obj))

        # === Weave display_name ===
        # wandb.display_name controls the UI display name
        display_name = metadata.get("display_name")
        if not display_name and model:
            if custom_llm_provider:
                display_name = f"{custom_llm_provider}/{model}"
            else:
                display_name = model
        if display_name:
            safe_set_attribute(span, "wandb.display_name", display_name)

        # Set token usage
        WeaveOtelLogger._set_token_usage(span=span, response_obj=response_obj)

        # Set response ID if available
        if response_obj and hasattr(response_obj, "get") and response_obj.get("id"):
            safe_set_attribute(span, "gen_ai.response.id", response_obj.get("id"))

    @staticmethod
    def _get_weave_host() -> str | None:
        """
        Returns the Weave OTEL host based on environment variables.

        Returned in the following order of precedence:
        1. WEAVE_OTEL_HOST
        2. WEAVE_HOST
        """
        return os.environ.get("WEAVE_OTEL_HOST") or os.environ.get("WEAVE_HOST")

    @staticmethod
    def get_weave_otel_config() -> WeaveOtelConfig:
        """
        Retrieves the Weave OpenTelemetry configuration based on environment variables.

        Environment Variables:
            WANDB_API_KEY: Required. W&B API key for authentication.
            WEAVE_PROJECT_ID: Required. Project ID in format <entity>/<project_name>.
            WEAVE_HOST: Optional. Custom Weave host URL. Defaults to cloud endpoint.

        Returns:
            WeaveOtelConfig: A Pydantic model containing Weave OTEL configuration.

        Raises:
            ValueError: If required environment variables are missing.
        """
        api_key = os.environ.get("WANDB_API_KEY", None)
        project_id = os.environ.get("WEAVE_PROJECT_ID", None)

        if not api_key:
            raise ValueError(
                "WANDB_API_KEY must be set for Weave OpenTelemetry integration."
            )

        if not project_id:
            raise ValueError(
                "WEAVE_PROJECT_ID must be set for Weave OpenTelemetry integration. "
                "Format: <entity>/<project_name>"
            )

        # Determine endpoint
        weave_host = WeaveOtelLogger._get_weave_host()

        if weave_host:
            if not weave_host.startswith("http"):
                weave_host = "https://" + weave_host
            # Self-managed instances use a different path
            endpoint = f"{weave_host.rstrip('/')}/traces/otel/v1/traces"
            verbose_logger.debug(f"Using Weave OTEL endpoint from host: {endpoint}")
        else:
            endpoint = WEAVE_CLOUD_ENDPOINT
            verbose_logger.debug(f"Using Weave cloud endpoint: {endpoint}")

        # Weave uses Basic auth with format: api:<WANDB_API_KEY>
        auth_header = WeaveOtelLogger._get_weave_authorization_header(api_key=api_key)

        # Weave requires project_id header
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

    @staticmethod
    def _get_weave_authorization_header(api_key: str) -> str:
        """
        Get the authorization header for Weave OpenTelemetry.

        Weave uses Basic auth with format: api:<WANDB_API_KEY>
        """
        auth_string = f"api:{api_key}"
        auth_header = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {auth_header}"

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
        dynamic_weave_project_id = standard_callback_dynamic_params.get(
            "weave_project_id"
        )

        if dynamic_wandb_api_key:
            auth_header = WeaveOtelLogger._get_weave_authorization_header(
                api_key=dynamic_wandb_api_key,
            )
            dynamic_headers["Authorization"] = auth_header

        if dynamic_weave_project_id:
            dynamic_headers["project_id"] = dynamic_weave_project_id

        return dynamic_headers if dynamic_headers else None
