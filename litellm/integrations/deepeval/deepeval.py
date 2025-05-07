import uuid

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.deepeval.api import Api, Endpoints, HttpMethods
from litellm.integrations.deepeval.types import BaseApiSpan, SpanApiType, TraceApi, TraceSpanApiStatus
from litellm.integrations.deepeval.utils import to_zod_compatible_iso
from litellm._logging import verbose_logger

# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class DeepEvalLogger(CustomLogger):
    """Logs litellm traces to DeepEval's platform."""

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a failure event to DeepEval's platform."""
        
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a success event to DeepEval's platform."""
        start_time = to_zod_compatible_iso(start_time)
        end_time = to_zod_compatible_iso(end_time)
        standard_logging_object = kwargs.get("standard_logging_object", None)
        
        if standard_logging_object:
            base_api_span = self._create_base_api_span(kwargs, standard_logging_object, start_time, end_time)
            trace_api = self._create_trace_api(base_api_span, standard_logging_object, start_time, end_time)

            try:
                body = trace_api.model_dump(
                    by_alias=True, exclude_none=True
                )
            except AttributeError:
                # Pydantic version below 2.0
                body = trace_api.dict(
                    by_alias=True, exclude_none=True
                )

            # Send the async request
            api = Api()
            response = await api.a_send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.TRACING_ENDPOINT,
                body=body,
            )

            verbose_logger.debug("DeepEvalLogger: async_log_success_event: Api response", response)

    
    def _create_base_api_span(self, kwargs, standard_logging_object, start_time, end_time):
        # extract usage
        usage = standard_logging_object.get("response", {}).get("usage", {})
        return BaseApiSpan(
            uuid = standard_logging_object.get("id", uuid.uuid4()),
            name = "litellm_success_callback",
            status = TraceSpanApiStatus.SUCCESS,
            type = SpanApiType.LLM,
            traceUuid = standard_logging_object.get("trace_id", uuid.uuid4()),
            startTime = str(start_time),
            endTime = str(end_time),
            input = kwargs.get("input", "NO_INPUT"),
            output = standard_logging_object.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", "NO_OUTPUT"),
            model=standard_logging_object.get("model", None),
            inputTokenCount=usage.get("prompt_tokens", None),
            outputTokenCount=usage.get("completion_tokens", None),
        )

    def _create_trace_api(self, base_api_span, standard_logging_object, start_time, end_time):
        return TraceApi(
            uuid = standard_logging_object.get("trace_id", uuid.uuid4()),
            baseSpans = [],
            agentSpans = [],
            llmSpans = [base_api_span],
            retrieverSpans = [],
            toolSpans = [],
            startTime = str(start_time),
            endTime = str(end_time),
        )

