import os
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm._uuid import uuid
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

_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return _STR_MAPPING.validate_python(value) if isinstance(value, dict) else None


def _as_list(value: object) -> list[object]:
    return _OBJECT_LIST.validate_python(value) if isinstance(value, list) else []


def _dict_items(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(mapping for item in _as_list(value) if (mapping := _as_mapping(item)) is not None)


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class DeepEvalLogger(CustomLogger):
    """Logs litellm traces to DeepEval's platform."""

    def __init__(self, *args, **kwargs):
        api_key: Final = os.getenv("CONFIDENT_API_KEY")
        self.litellm_environment = os.getenv("LITELM_ENVIRONMENT", "development")
        validate_environment(self.litellm_environment)
        if not api_key:
            raise ValueError("Please set 'CONFIDENT_API_KEY=<>' in your environment variables.")
        self.api = Api(api_key=api_key)
        super().__init__(*args, **kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a success event to DeepEval's platform."""
        self._sync_event_handler(kwargs, response_obj, start_time, end_time, is_success=True)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a failure event to DeepEval's platform."""
        self._sync_event_handler(kwargs, response_obj, start_time, end_time, is_success=False)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a failure event to DeepEval's platform."""
        await self._async_event_handler(kwargs, response_obj, start_time, end_time, is_success=False)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logs a success event to DeepEval's platform."""
        await self._async_event_handler(kwargs, response_obj, start_time, end_time, is_success=True)

    def _prepare_trace_api(self, kwargs, response_obj, start_time, end_time, is_success):
        _start_time: Final = to_zod_compatible_iso(start_time)
        _end_time: Final = to_zod_compatible_iso(end_time)
        _standard_logging_object: Final = kwargs.get("standard_logging_object", {})
        base_api_span: Final = self._create_base_api_span(
            kwargs,
            standard_logging_object=_standard_logging_object,
            start_time=_start_time,
            end_time=_end_time,
            is_success=is_success,
        )
        trace_api: Final = self._create_trace_api(
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

    def _sync_event_handler(self, kwargs, response_obj, start_time, end_time, is_success):
        body: Final = self._prepare_trace_api(kwargs, response_obj, start_time, end_time, is_success)
        try:
            response: Final = self.api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.TRACING_ENDPOINT,
                body=body,
            )
        except Exception as e:
            raise e
        verbose_logger.debug("DeepEvalLogger: sync_log_failure_event: Api response %s", response)

    async def _async_event_handler(self, kwargs, response_obj, start_time, end_time, is_success):
        body: Final = self._prepare_trace_api(kwargs, response_obj, start_time, end_time, is_success)
        response: Final = await self.api.a_send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.TRACING_ENDPOINT,
            body=body,
        )

        verbose_logger.debug("DeepEvalLogger: async_event_handler: Api response %s", response)

    @staticmethod
    def _get_success_output(response: object) -> str | list[object] | None:
        """Chat completions read ``choices``; Responses API payloads read ``output``."""
        response_dict: Final = _as_mapping(response) or {}
        choices: Final = _dict_items(response_dict.get("choices"))
        if choices:
            message: Final = _as_mapping(choices[0].get("message")) or {}
            content: Final = message.get("content")
            tool_calls: Final = _as_list(message.get("tool_calls"))
            if content:
                return content
            if tool_calls:
                return tool_calls
            if isinstance(content, str):
                return content
            return _as_list(content) or None
        output_items: Final = _as_list(response_dict.get("output"))
        if output_items:
            output_texts: Final = tuple(
                text
                for item in _dict_items(output_items)
                if item.get("type") == "message"
                for part in _dict_items(item.get("content"))
                if part.get("type") == "output_text"
                if isinstance(text := part.get("text"), str)
            )
            return "".join(output_texts) if output_texts else output_items
        return "NO_OUTPUT"

    def _create_base_api_span(self, kwargs, standard_logging_object, start_time, end_time, is_success):
        # extract usage
        usage: Final = standard_logging_object.get("response", {}).get("usage", {})
        if is_success:
            output = self._get_success_output(standard_logging_object.get("response", {}))
        else:
            output = str(standard_logging_object.get("error_string", ""))
        return BaseApiSpan(
            uuid=standard_logging_object.get("id", uuid.uuid4()),
            name=("litellm_success_callback" if is_success else "litellm_failure_callback"),
            status=(TraceSpanApiStatus.SUCCESS if is_success else TraceSpanApiStatus.ERRORED),
            type=SpanApiType.LLM,
            traceUuid=standard_logging_object.get("trace_id", uuid.uuid4()),
            startTime=str(start_time),
            endTime=str(end_time),
            input=kwargs.get("input", "NO_INPUT"),
            output=output,
            model=standard_logging_object.get("model", None),
            inputTokenCount=usage.get("prompt_tokens", None) if is_success else None,
            outputTokenCount=(usage.get("completion_tokens", None) if is_success else None),
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
