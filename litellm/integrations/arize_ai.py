"""
arize AI is OTEL compatible

this file has Arize ai specific helper functions
"""

import json
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from .opentelemetry import OpenTelemetryConfig as _OpenTelemetryConfig

    Span = _Span
    OpenTelemetryConfig = _OpenTelemetryConfig
else:
    Span = Any
    OpenTelemetryConfig = Any

import os

from litellm.types.integrations.arize import *


class ArizeLogger:
    @staticmethod
    def set_arize_ai_attributes(span: Span, kwargs, response_obj):
        from litellm.integrations._types.open_inference import (
            MessageAttributes,
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )

        try:

            optional_params = kwargs.get("optional_params", {})
            # litellm_params = kwargs.get("litellm_params", {}) or {}

            #############################################
            ############ LLM CALL METADATA ##############
            #############################################
            # commented out for now - looks like Arize AI could not log this
            # metadata = litellm_params.get("metadata", {}) or {}
            # span.set_attribute(SpanAttributes.METADATA, str(metadata))

            #############################################
            ########## LLM Request Attributes ###########
            #############################################

            # The name of the LLM a request is being made to
            if kwargs.get("model"):
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, kwargs.get("model"))

            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.LLM.value,
            )
            messages = kwargs.get("messages")

            # for /chat/completions
            # https://docs.arize.com/arize/large-language-models/tracing/semantic-conventions
            if messages:
                span.set_attribute(
                    SpanAttributes.INPUT_VALUE,
                    messages[-1].get("content", ""),  # get the last message for input
                )

                # LLM_INPUT_MESSAGES shows up under `input_messages` tab on the span page
                for idx, msg in enumerate(messages):
                    # Set the role per message
                    span.set_attribute(
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_ROLE}",
                        msg["role"],
                    )
                    # Set the content per message
                    span.set_attribute(
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_CONTENT}",
                        msg.get("content", ""),
                    )

            # The Generative AI Provider: Azure, OpenAI, etc.
            _optional_params = ArizeLogger.make_json_serializable(optional_params)
            _json_optional_params = json.dumps(_optional_params)
            span.set_attribute(
                SpanAttributes.LLM_INVOCATION_PARAMETERS, _json_optional_params
            )

            if optional_params.get("user"):
                span.set_attribute(SpanAttributes.USER_ID, optional_params.get("user"))

            #############################################
            ########## LLM Response Attributes ##########
            # https://docs.arize.com/arize/large-language-models/tracing/semantic-conventions
            #############################################
            for choice in response_obj.get("choices"):
                response_message = choice.get("message", {})
                span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, response_message.get("content", "")
                )

                # This shows up under `output_messages` tab on the span page
                # This code assumes a single response
                span.set_attribute(
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
                    response_message["role"],
                )
                span.set_attribute(
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
                    response_message.get("content", ""),
                )

            usage = response_obj.get("usage")
            if usage:
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    usage.get("total_tokens"),
                )

                # The number of tokens used in the LLM response (completion).
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    usage.get("completion_tokens"),
                )

                # The number of tokens used in the LLM prompt.
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    usage.get("prompt_tokens"),
                )
            pass
        except Exception as e:
            verbose_logger.error(f"Error setting arize attributes: {e}")

    ###################### Helper functions ######################

    @staticmethod
    def _get_arize_config() -> ArizeConfig:
        """
        Helper function to get Arize configuration.

        Returns:
            ArizeConfig: A Pydantic model containing Arize configuration.

        Raises:
            ValueError: If required environment variables are not set.
        """
        space_key = os.environ.get("ARIZE_SPACE_KEY")
        api_key = os.environ.get("ARIZE_API_KEY")

        if not space_key:
            raise ValueError("ARIZE_SPACE_KEY not found in environment variables")
        if not api_key:
            raise ValueError("ARIZE_API_KEY not found in environment variables")

        grpc_endpoint = os.environ.get("ARIZE_ENDPOINT")
        http_endpoint = os.environ.get("ARIZE_HTTP_ENDPOINT")
        if grpc_endpoint is None and http_endpoint is None:
            # use default arize grpc endpoint
            verbose_logger.debug(
                "No ARIZE_ENDPOINT or ARIZE_HTTP_ENDPOINT found, using default endpoint: https://otlp.arize.com/v1"
            )
            grpc_endpoint = "https://otlp.arize.com/v1"

        return ArizeConfig(
            space_key=space_key,
            api_key=api_key,
            grpc_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
        )

    @staticmethod
    def get_arize_opentelemetry_config() -> Optional[OpenTelemetryConfig]:
        """
        Helper function to get OpenTelemetry configuration for Arize.

        Args:
            arize_config (ArizeConfig): Arize configuration object.

        Returns:
            OpenTelemetryConfig: Configuration for OpenTelemetry.
        """
        from .opentelemetry import OpenTelemetryConfig

        arize_config = ArizeLogger._get_arize_config()
        if arize_config.http_endpoint:
            return OpenTelemetryConfig(
                exporter="otlp_http",
                endpoint=arize_config.http_endpoint,
            )

        # use default arize grpc endpoint
        return OpenTelemetryConfig(
            exporter="otlp_grpc",
            endpoint=arize_config.grpc_endpoint,
        )

    @staticmethod
    def make_json_serializable(payload: dict) -> dict:
        for key, value in payload.items():
            try:
                if isinstance(value, dict):
                    # recursively sanitize dicts
                    payload[key] = ArizeLogger.make_json_serializable(value.copy())
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    # everything else becomes a string
                    payload[key] = str(value)
            except Exception:
                # non blocking if it can't cast to a str
                pass
        return payload
