import os
import uuid
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.deepeval.api import Api, Endpoints, HttpMethods
from litellm.integrations.deepeval.types import (
    BaseApiSpan,
    SpanApiType,
    TraceApi,
    TraceSpanApiStatus,
)
from litellm.integrations.deepeval.utils import (
    to_zod_compatible_iso,
    validate_environment,
)
from litellm._logging import verbose_logger


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class DeepEvalLogger(CustomLogger):
    """Logs litellm traces to DeepEval's platform."""

    def __init__(self, *args, **kwargs):
        api_key = os.getenv("CONFIDENT_API_KEY")
        self.litellm_environment = os.getenv("LITELM_ENVIRONMENT", "development")
        validate_environment(self.litellm_environment)
        if not api_key:
            raise ValueError(
                "Please set 'CONFIDENT_API_KEY=<>' in your environment variables."
            )
        self.api = Api(api_key=api_key)
        super().__init__(*args, **kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a success event to DeepEval's platform."""
        self._sync_event_handler(
            kwargs, response_obj, start_time, end_time, is_success=True
        )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a failure event to DeepEval's platform."""
        self._sync_event_handler(
            kwargs, response_obj, start_time, end_time, is_success=False
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a failure event to DeepEval's platform."""
        await self._async_event_handler(
            kwargs, response_obj, start_time, end_time, is_success=False
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a success event to DeepEval's platform."""
        await self._async_event_handler(
            kwargs, response_obj, start_time, end_time, is_success=True
        )

    def _prepare_trace_api(
        self, kwargs, response_obj, start_time, end_time, is_success
    ):
        _start_time = to_zod_compatible_iso(start_time)
        _end_time = to_zod_compatible_iso(end_time)
        _standard_logging_object = kwargs.get("standard_logging_object", {})
        base_api_span = self._create_base_api_span(
            kwargs,
            standard_logging_object=_standard_logging_object,
            start_time=_start_time,
            end_time=_end_time,
            is_success=is_success,
        )
        trace_api = self._create_trace_api(
            base_api_span,
            standard_logging_object=_standard_logging_object,
            start_time=_start_time,
            end_time=_end_time,
            litellm_environment=self.litellm_environment,
        )

        body = {}

        try:
            body = trace_api.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = trace_api.dict(by_alias=True, exclude_none=True)
        return body

    def _sync_event_handler(
        self, kwargs, response_obj, start_time, end_time, is_success
    ):
        body = self._prepare_trace_api(
            kwargs, response_obj, start_time, end_time, is_success
        )
        try:
            response = self.api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.TRACING_ENDPOINT,
                body=body,
            )
        except Exception as e:
            raise e
        verbose_logger.debug(
            "DeepEvalLogger: sync_log_failure_event: Api response", response
        )

    async def _async_event_handler(
        self, kwargs, response_obj, start_time, end_time, is_success
    ):
        body = self._prepare_trace_api(
            kwargs, response_obj, start_time, end_time, is_success
        )
        response = await self.api.a_send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.TRACING_ENDPOINT,
            body=body,
        )

        verbose_logger.debug(
            "DeepEvalLogger: async_event_handler: Api response", response
        )

    def _create_base_api_span(
        self, kwargs, standard_logging_object, start_time, end_time, is_success
    ):
        # extract usage
        usage = standard_logging_object.get("response", {}).get("usage", {})
        if is_success:
            output = (
                standard_logging_object.get("response", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "NO_OUTPUT")
            )
        else:
            output = str(standard_logging_object.get("error_string", ""))
        return BaseApiSpan(
            uuid=standard_logging_object.get("id", uuid.uuid4()),
            name=(
                "litellm_success_callback" if is_success else "litellm_failure_callback"
            ),
            status=(
                TraceSpanApiStatus.SUCCESS if is_success else TraceSpanApiStatus.ERRORED
            ),
            type=SpanApiType.LLM,
            traceUuid=standard_logging_object.get("trace_id", uuid.uuid4()),
            startTime=str(start_time),
            endTime=str(end_time),
            input=kwargs.get("input", "NO_INPUT"),
            output=output,
            model=standard_logging_object.get("model", None),
            inputTokenCount=usage.get("prompt_tokens", None) if is_success else None,
            outputTokenCount=(
                usage.get("completion_tokens", None) if is_success else None
            ),
        )

    def _create_trace_api(
        self,
        base_api_span,
        standard_logging_object,
        start_time,
        end_time,
        litellm_environment,
    ):
        return TraceApi(
            uuid=standard_logging_object.get("trace_id", uuid.uuid4()),
            baseSpans=[],
            agentSpans=[],
            llmSpans=[base_api_span],
            retrieverSpans=[],
            toolSpans=[],
            startTime=str(start_time),
            endTime=str(end_time),
            environment=litellm_environment,
        )
