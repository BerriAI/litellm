import os
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.services import ServiceLoggerPayload
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy._types import (
        ManagementEndpointLoggingPayload as _ManagementEndpointLoggingPayload,
    )
    from litellm.proxy.proxy_server import UserAPIKeyAuth as _UserAPIKeyAuth

    Span = _Span
    UserAPIKeyAuth = _UserAPIKeyAuth
    ManagementEndpointLoggingPayload = _ManagementEndpointLoggingPayload
else:
    Span = Any
    UserAPIKeyAuth = Any
    ManagementEndpointLoggingPayload = Any


LITELLM_TRACER_NAME = os.getenv("OTEL_TRACER_NAME", "litellm")
LITELLM_RESOURCE: Dict[Any, Any] = {
    "service.name": os.getenv("OTEL_SERVICE_NAME", "litellm"),
    "deployment.environment": os.getenv("OTEL_ENVIRONMENT_NAME", "production"),
    "model_id": os.getenv("OTEL_SERVICE_NAME", "litellm"),
}
RAW_REQUEST_SPAN_NAME = "raw_gen_ai_request"
LITELLM_REQUEST_SPAN_NAME = "litellm_request"


@dataclass
class OpenTelemetryConfig:
    from opentelemetry.sdk.trace.export import SpanExporter

    exporter: Union[str, SpanExporter] = "console"
    endpoint: Optional[str] = None
    headers: Optional[str] = None

    @classmethod
    def from_env(cls):
        """
        OTEL_HEADERS=x-honeycomb-team=B85YgLm9****
        OTEL_EXPORTER="otlp_http"
        OTEL_ENDPOINT="https://api.honeycomb.io/v1/traces"

        OTEL_HEADERS gets sent as headers = {"x-honeycomb-team": "B85YgLm96******"}
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        if os.getenv("OTEL_EXPORTER") == "in_memory":
            return cls(exporter=InMemorySpanExporter())
        return cls(
            exporter=os.getenv("OTEL_EXPORTER", "console"),
            endpoint=os.getenv("OTEL_ENDPOINT"),
            headers=os.getenv(
                "OTEL_HEADERS"
            ),  # example: OTEL_HEADERS=x-honeycomb-team=B85YgLm96VGdFisfJVme1H"
        )


class OpenTelemetry(CustomLogger):
    def __init__(
        self,
        config: OpenTelemetryConfig = OpenTelemetryConfig.from_env(),
        callback_name: Optional[str] = None,
        **kwargs,
    ):
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        self.config = config
        self.OTEL_EXPORTER = self.config.exporter
        self.OTEL_ENDPOINT = self.config.endpoint
        self.OTEL_HEADERS = self.config.headers
        provider = TracerProvider(resource=Resource(attributes=LITELLM_RESOURCE))
        provider.add_span_processor(self._get_span_processor())
        self.callback_name = callback_name

        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(LITELLM_TRACER_NAME)

        _debug_otel = str(os.getenv("DEBUG_OTEL", "False")).lower()

        if _debug_otel == "true":
            # Set up logging
            import logging

            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger(__name__)

            # Enable OpenTelemetry logging
            otel_exporter_logger = logging.getLogger("opentelemetry.sdk.trace.export")
            otel_exporter_logger.setLevel(logging.DEBUG)

        # init CustomLogger params
        super().__init__(**kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_sucess(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_sucess(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_service_success_hook(
        self,
        payload: ServiceLoggerPayload,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ):
        from datetime import datetime

        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        if isinstance(start_time, float):
            _start_time_ns = int(int(start_time) * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(int(end_time) * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = payload.service
            service_logging_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="call_type",
                value=payload.call_type,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="service",
                value=payload.service.value,
            )

            if event_metadata:
                for key, value in event_metadata.items():
                    if value is None:
                        value = "None"
                    if isinstance(value, dict):
                        try:
                            value = str(value)
                        except Exception:
                            value = "litellm logging error - could_not_json_serialize"
                    self.safe_set_attribute(
                        span=service_logging_span,
                        key=key,
                        value=value,
                    )
            service_logging_span.set_status(Status(StatusCode.OK))
            service_logging_span.end(end_time=_end_time_ns)

    async def async_service_failure_hook(
        self,
        payload: ServiceLoggerPayload,
        error: Optional[str] = "",
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[float, datetime]] = None,
        event_metadata: Optional[dict] = None,
    ):
        from datetime import datetime

        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        if isinstance(start_time, float):
            _start_time_ns = int(int(start_time) * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(int(end_time) * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = payload.service
            service_logging_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="call_type",
                value=payload.call_type,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="service",
                value=payload.service.value,
            )
            if error:
                self.safe_set_attribute(
                    span=service_logging_span,
                    key="error",
                    value=error,
                )
            if event_metadata:
                for key, value in event_metadata.items():
                    if isinstance(value, dict):
                        try:
                            value = str(value)
                        except Exception:
                            value = "litllm logging error - could_not_json_serialize"
                    self.safe_set_attribute(
                        span=service_logging_span,
                        key=key,
                        value=value,
                    )

            service_logging_span.set_status(Status(StatusCode.ERROR))
            service_logging_span.end(end_time=_end_time_ns)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        parent_otel_span = user_api_key_dict.parent_otel_span
        if parent_otel_span is not None:
            parent_otel_span.set_status(Status(StatusCode.ERROR))
            _span_name = "Failed Proxy Server Request"

            # Exception Logging Child Span
            exception_logging_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
            )
            self.safe_set_attribute(
                span=exception_logging_span,
                key="exception",
                value=str(original_exception),
            )
            exception_logging_span.set_status(Status(StatusCode.ERROR))
            exception_logging_span.end(end_time=self._to_ns(datetime.now()))

            # End Parent OTEL Sspan
            parent_otel_span.end(end_time=self._to_ns(datetime.now()))

    def _handle_sucess(self, kwargs, response_obj, start_time, end_time):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        verbose_logger.debug(
            "OpenTelemetry Logger: Logging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )
        _parent_context, parent_otel_span = self._get_span_context(kwargs)

        # Span 1: Requst sent to litellm SDK
        span = self.tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=_parent_context,
        )
        span.set_status(Status(StatusCode.OK))
        self.set_attributes(span, kwargs, response_obj)

        if litellm.turn_off_message_logging is True:
            pass
        elif self.message_logging is not True:
            pass
        else:
            # Span 2: Raw Request / Response to LLM
            raw_request_span = self.tracer.start_span(
                name=RAW_REQUEST_SPAN_NAME,
                start_time=self._to_ns(start_time),
                context=trace.set_span_in_context(span),
            )

            raw_request_span.set_status(Status(StatusCode.OK))
            self.set_raw_request_attributes(raw_request_span, kwargs, response_obj)
            raw_request_span.end(end_time=self._to_ns(end_time))

        span.end(end_time=self._to_ns(end_time))

        if parent_otel_span is not None:
            parent_otel_span.end(end_time=self._to_ns(datetime.now()))

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        from opentelemetry.trace import Status, StatusCode

        verbose_logger.debug(
            "OpenTelemetry Logger: Failure HandlerLogging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )
        _parent_context, parent_otel_span = self._get_span_context(kwargs)

        # Span 1: Requst sent to litellm SDK
        span = self.tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=_parent_context,
        )
        span.set_status(Status(StatusCode.ERROR))
        self.set_attributes(span, kwargs, response_obj)
        span.end(end_time=self._to_ns(end_time))

        if parent_otel_span is not None:
            parent_otel_span.end(end_time=self._to_ns(datetime.now()))

    def set_tools_attributes(self, span: Span, tools):
        import json

        from litellm.proxy._types import SpanAttributes

        if not tools:
            return

        try:
            for i, tool in enumerate(tools):
                function = tool.get("function")
                if not function:
                    continue

                prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
                self.safe_set_attribute(
                    span=span,
                    key=f"{prefix}.name",
                    value=function.get("name"),
                )
                self.safe_set_attribute(
                    span=span,
                    key=f"{prefix}.description",
                    value=function.get("description"),
                )
                self.safe_set_attribute(
                    span=span,
                    key=f"{prefix}.parameters",
                    value=json.dumps(function.get("parameters")),
                )
        except Exception as e:
            verbose_logger.error(
                "OpenTelemetry: Error setting tools attributes: %s", str(e)
            )
            pass

    def cast_as_primitive_value_type(self, value) -> Union[str, bool, int, float]:
        """
        Casts the value to a primitive OTEL type if it is not already a primitive type.

        OTEL supports - str, bool, int, float

        If it's not a primitive type, then it's converted to a string
        """
        if value is None:
            return ""
        if isinstance(value, (str, bool, int, float)):
            return value
        try:
            return str(value)
        except Exception:
            return ""

    def set_attributes(self, span: Span, kwargs, response_obj):  # noqa: PLR0915
        try:
            if self.callback_name == "arize":
                from litellm.integrations.arize_ai import ArizeLogger

                ArizeLogger.set_arize_ai_attributes(span, kwargs, response_obj)
                return
            elif self.callback_name == "langtrace":
                from litellm.integrations.langtrace import LangtraceAttributes

                LangtraceAttributes().set_langtrace_attributes(
                    span, kwargs, response_obj
                )
                return
            from litellm.proxy._types import SpanAttributes

            optional_params = kwargs.get("optional_params", {})
            litellm_params = kwargs.get("litellm_params", {}) or {}
            standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")

            # https://github.com/open-telemetry/semantic-conventions/blob/main/model/registry/gen-ai.yaml
            # Following Conventions here: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/llm-spans.md
            #############################################
            ############ LLM CALL METADATA ##############
            #############################################
            metadata = standard_logging_payload["metadata"]
            for key, value in metadata.items():
                self.safe_set_attribute(
                    span=span, key="metadata.{}".format(key), value=value
                )

            #############################################
            ########## LLM Request Attributes ###########
            #############################################

            # The name of the LLM a request is being made to
            if kwargs.get("model"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_MODEL,
                    value=kwargs.get("model"),
                )

            # The Generative AI Provider: Azure, OpenAI, etc.
            self.safe_set_attribute(
                span=span,
                key=SpanAttributes.LLM_SYSTEM,
                value=litellm_params.get("custom_llm_provider", "Unknown"),
            )

            # The maximum number of tokens the LLM generates for a request.
            if optional_params.get("max_tokens"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_MAX_TOKENS,
                    value=optional_params.get("max_tokens"),
                )

            # The temperature setting for the LLM request.
            if optional_params.get("temperature"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_TEMPERATURE,
                    value=optional_params.get("temperature"),
                )

            # The top_p sampling setting for the LLM request.
            if optional_params.get("top_p"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_TOP_P,
                    value=optional_params.get("top_p"),
                )

            self.safe_set_attribute(
                span=span,
                key=SpanAttributes.LLM_IS_STREAMING,
                value=str(optional_params.get("stream", False)),
            )

            if optional_params.get("user"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_USER,
                    value=optional_params.get("user"),
                )

            # The unique identifier for the completion.
            if response_obj.get("id"):
                self.safe_set_attribute(
                    span=span, key="gen_ai.response.id", value=response_obj.get("id")
                )

            # The model used to generate the response.
            if response_obj.get("model"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_RESPONSE_MODEL,
                    value=response_obj.get("model"),
                )

            usage = response_obj.get("usage")
            if usage:
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                    value=usage.get("total_tokens"),
                )

                # The number of tokens used in the LLM response (completion).
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                    value=usage.get("completion_tokens"),
                )

                # The number of tokens used in the LLM prompt.
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                    value=usage.get("prompt_tokens"),
                )

            ########################################################################
            ########## LLM Request Medssages / tools / content Attributes ###########
            #########################################################################

            if litellm.turn_off_message_logging is True:
                return
            if self.message_logging is not True:
                return

            if optional_params.get("tools"):
                tools = optional_params["tools"]
                self.set_tools_attributes(span, tools)

            if kwargs.get("messages"):
                for idx, prompt in enumerate(kwargs.get("messages")):
                    if prompt.get("role"):
                        self.safe_set_attribute(
                            span=span,
                            key=f"{SpanAttributes.LLM_PROMPTS}.{idx}.role",
                            value=prompt.get("role"),
                        )

                    if prompt.get("content"):
                        if not isinstance(prompt.get("content"), str):
                            prompt["content"] = str(prompt.get("content"))
                        self.safe_set_attribute(
                            span=span,
                            key=f"{SpanAttributes.LLM_PROMPTS}.{idx}.content",
                            value=prompt.get("content"),
                        )
            #############################################
            ########## LLM Response Attributes ##########
            #############################################
            if response_obj is not None:
                if response_obj.get("choices"):
                    for idx, choice in enumerate(response_obj.get("choices")):
                        if choice.get("finish_reason"):
                            self.safe_set_attribute(
                                span=span,
                                key=f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.finish_reason",
                                value=choice.get("finish_reason"),
                            )
                        if choice.get("message"):
                            if choice.get("message").get("role"):
                                self.safe_set_attribute(
                                    span=span,
                                    key=f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role",
                                    value=choice.get("message").get("role"),
                                )
                            if choice.get("message").get("content"):
                                if not isinstance(
                                    choice.get("message").get("content"), str
                                ):
                                    choice["message"]["content"] = str(
                                        choice.get("message").get("content")
                                    )
                                self.safe_set_attribute(
                                    span=span,
                                    key=f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content",
                                    value=choice.get("message").get("content"),
                                )

                            message = choice.get("message")
                            tool_calls = message.get("tool_calls")
                            if tool_calls:
                                self.safe_set_attribute(
                                    span=span,
                                    key=f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.function_call.name",
                                    value=tool_calls[0].get("function").get("name"),
                                )
                                self.safe_set_attribute(
                                    span=span,
                                    key=f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.function_call.arguments",
                                    value=tool_calls[0]
                                    .get("function")
                                    .get("arguments"),
                                )

        except Exception as e:
            verbose_logger.error(
                "OpenTelemetry logging error in set_attributes %s", str(e)
            )

    def _cast_as_primitive_value_type(self, value) -> Union[str, bool, int, float]:
        """
        Casts the value to a primitive OTEL type if it is not already a primitive type.

        OTEL supports - str, bool, int, float

        If it's not a primitive type, then it's converted to a string
        """
        if value is None:
            return ""
        if isinstance(value, (str, bool, int, float)):
            return value
        try:
            return str(value)
        except Exception:
            return ""

    def safe_set_attribute(self, span: Span, key: str, value: Any):
        """
        Safely sets an attribute on the span, ensuring the value is a primitive type.
        """
        primitive_value = self._cast_as_primitive_value_type(value)
        span.set_attribute(key, primitive_value)

    def set_raw_request_attributes(self, span: Span, kwargs, response_obj):
        from litellm.proxy._types import SpanAttributes

        kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {}) or {}
        custom_llm_provider = litellm_params.get("custom_llm_provider", "Unknown")

        _raw_response = kwargs.get("original_response")
        _additional_args = kwargs.get("additional_args", {}) or {}
        complete_input_dict = _additional_args.get("complete_input_dict")
        #############################################
        ########## LLM Request Attributes ###########
        #############################################

        # OTEL Attributes for the RAW Request to https://docs.anthropic.com/en/api/messages
        if complete_input_dict and isinstance(complete_input_dict, dict):
            for param, val in complete_input_dict.items():
                self.safe_set_attribute(
                    span=span, key=f"llm.{custom_llm_provider}.{param}", value=val
                )

        #############################################
        ########## LLM Response Attributes ##########
        #############################################
        if _raw_response and isinstance(_raw_response, str):
            # cast sr -> dict
            import json

            try:
                _raw_response = json.loads(_raw_response)
                for param, val in _raw_response.items():
                    self.safe_set_attribute(
                        span=span,
                        key=f"llm.{custom_llm_provider}.{param}",
                        value=val,
                    )
            except json.JSONDecodeError:
                verbose_logger.debug(
                    "litellm.integrations.opentelemetry.py::set_raw_request_attributes() - raw_response not json string - {}".format(
                        _raw_response
                    )
                )

                self.safe_set_attribute(
                    span=span,
                    key=f"llm.{custom_llm_provider}.stringified_raw_response",
                    value=_raw_response,
                )

    def _to_ns(self, dt):
        return int(dt.timestamp() * 1e9)

    def _get_span_name(self, kwargs):
        return LITELLM_REQUEST_SPAN_NAME

    def get_traceparent_from_header(self, headers):
        if headers is None:
            return None
        _traceparent = headers.get("traceparent", None)
        if _traceparent is None:
            return None

        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        verbose_logger.debug("OpenTelemetry: GOT A TRACEPARENT {}".format(_traceparent))
        propagator = TraceContextTextMapPropagator()
        _parent_context = propagator.extract(carrier={"traceparent": _traceparent})
        verbose_logger.debug("OpenTelemetry: PARENT CONTEXT {}".format(_parent_context))
        return _parent_context

    def _get_span_context(self, kwargs):
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request", {}) or {}
        headers = proxy_server_request.get("headers", {}) or {}
        traceparent = headers.get("traceparent", None)
        _metadata = litellm_params.get("metadata", {}) or {}
        parent_otel_span = _metadata.get("litellm_parent_otel_span", None)

        """
        Two way to use parents in opentelemetry
        - using the traceparent header
        - using the parent_otel_span in the [metadata][parent_otel_span]
        """
        if parent_otel_span is not None:
            return trace.set_span_in_context(parent_otel_span), parent_otel_span

        if traceparent is None:
            return None, None
        else:
            carrier = {"traceparent": traceparent}
            return TraceContextTextMapPropagator().extract(carrier=carrier), None

    def _get_span_processor(self):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
            SpanExporter,
        )

        verbose_logger.debug(
            "OpenTelemetry Logger, initializing span processor \nself.OTEL_EXPORTER: %s\nself.OTEL_ENDPOINT: %s\nself.OTEL_HEADERS: %s",
            self.OTEL_EXPORTER,
            self.OTEL_ENDPOINT,
            self.OTEL_HEADERS,
        )
        _split_otel_headers = {}
        if self.OTEL_HEADERS is not None and isinstance(self.OTEL_HEADERS, str):
            _split_otel_headers = self.OTEL_HEADERS.split("=")
            _split_otel_headers = {_split_otel_headers[0]: _split_otel_headers[1]}

        if isinstance(self.OTEL_EXPORTER, SpanExporter):
            verbose_logger.debug(
                "OpenTelemetry: intiializing SpanExporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return SimpleSpanProcessor(self.OTEL_EXPORTER)

        if self.OTEL_EXPORTER == "console":
            verbose_logger.debug(
                "OpenTelemetry: intiializing console exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(ConsoleSpanExporter())
        elif self.OTEL_EXPORTER == "otlp_http":
            verbose_logger.debug(
                "OpenTelemetry: intiializing http exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(
                OTLPSpanExporterHTTP(
                    endpoint=self.OTEL_ENDPOINT, headers=_split_otel_headers
                ),
            )
        elif self.OTEL_EXPORTER == "otlp_grpc":
            verbose_logger.debug(
                "OpenTelemetry: intiializing grpc exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(
                OTLPSpanExporterGRPC(
                    endpoint=self.OTEL_ENDPOINT, headers=_split_otel_headers
                ),
            )
        else:
            verbose_logger.debug(
                "OpenTelemetry: intiializing console exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(ConsoleSpanExporter())

    async def async_management_endpoint_success_hook(
        self,
        logging_payload: ManagementEndpointLoggingPayload,
        parent_otel_span: Optional[Span] = None,
    ):
        from datetime import datetime

        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        start_time = logging_payload.start_time
        end_time = logging_payload.end_time

        if isinstance(start_time, float):
            _start_time_ns = int(int(start_time) * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(int(end_time) * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = logging_payload.route
            management_endpoint_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )

            _request_data = logging_payload.request_data
            if _request_data is not None:
                for key, value in _request_data.items():
                    self.safe_set_attribute(
                        span=management_endpoint_span,
                        key=f"request.{key}",
                        value=value,
                    )

            _response = logging_payload.response
            if _response is not None:
                for key, value in _response.items():
                    self.safe_set_attribute(
                        span=management_endpoint_span,
                        key=f"response.{key}",
                        value=value,
                    )

            management_endpoint_span.set_status(Status(StatusCode.OK))
            management_endpoint_span.end(end_time=_end_time_ns)

    async def async_management_endpoint_failure_hook(
        self,
        logging_payload: ManagementEndpointLoggingPayload,
        parent_otel_span: Optional[Span] = None,
    ):
        from datetime import datetime

        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        start_time = logging_payload.start_time
        end_time = logging_payload.end_time

        if isinstance(start_time, float):
            _start_time_ns = int(int(start_time) * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(int(end_time) * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = logging_payload.route
            management_endpoint_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )

            _request_data = logging_payload.request_data
            if _request_data is not None:
                for key, value in _request_data.items():
                    self.safe_set_attribute(
                        span=management_endpoint_span,
                        key=f"request.{key}",
                        value=value,
                    )

            _exception = logging_payload.exception
            self.safe_set_attribute(
                span=management_endpoint_span,
                key="exception",
                value=str(_exception),
            )
            management_endpoint_span.set_status(Status(StatusCode.ERROR))
            management_endpoint_span.end(end_time=_end_time_ns)
