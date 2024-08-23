import traceback
import json
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import SpanAttributes


class LangtraceLogger(CustomLogger):
    """
    This class is used to log traces to Langtrace
    """

    def __init__(self) -> None:
        try:
            from langtrace_python_sdk import langtrace
            from opentelemetry.trace import get_tracer

        except ModuleNotFoundError as e:
            raise Exception(
                f"\033[91mLangtrace not installed, try running 'pip install langtrace-python-sdk' to fix this error: {e}\n{traceback.format_exc()}\033[0m"
            )

        self.tracer = get_tracer(__name__)

    def get_timestamps_ns(self, start_time, end_time):
        """
        This function is used to get the timestamps in nanoseconds
        """
        _start_time_ns = 0
        _end_time_ns = 0

        if isinstance(start_time, float):
            _start_time_ns = int(int(start_time) * 1e9)
        else:
            _start_time_ns = int(start_time.timestamp() * 1e9)

        if isinstance(end_time, float):
            _end_time_ns = int(int(end_time) * 1e9)
        else:
            _end_time_ns = int(end_time.timestamp() * 1e9)

        return _start_time_ns, _end_time_ns

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        """
        This function is used to log the event to Langtrace
        """
        try:
            from opentelemetry.trace import SpanKind

            _start_time_ns, _end_time_ns = self.get_timestamps_ns(start_time, end_time)
            vendor = kwargs.get("litellm_params").get("custom_llm_provider")
            span = self.tracer.start_span(
                name=f"chat {kwargs.get('model')}",
                kind=SpanKind.CLIENT,
                start_time=_start_time_ns,
            )

            optional_params = kwargs.get("optional_params", {})
            options = {**kwargs, **optional_params}
            self.set_request_attributes(span, options, vendor)
            self.set_response_attributes(span, response_obj)
            self.set_usage_attributes(span, response_obj)

            span.end(end_time=_end_time_ns)
        except Exception as e:
            print_verbose(f"LangtraceLogger Error - {traceback.format_exc()}")

    def set_request_attributes(self, span, kwargs, vendor):
        """
        This function is used to get span attributes for the LLM request
        """
        span_attributes = {
            "gen_ai.operation.name": "chat",
            "langtrace.service.name": vendor,
            SpanAttributes.LLM_REQUEST_MODEL.value: kwargs.get("model"),
            SpanAttributes.LLM_IS_STREAMING.value: kwargs.get("stream"),
            SpanAttributes.LLM_REQUEST_TEMPERATURE.value: kwargs.get("temperature"),
            SpanAttributes.LLM_TOP_K.value: kwargs.get("top_k"),
            SpanAttributes.LLM_REQUEST_TOP_P.value: kwargs.get("top_p"),
            SpanAttributes.LLM_USER.value: kwargs.get("user"),
            SpanAttributes.LLM_REQUEST_MAX_TOKENS.value: kwargs.get("max_tokens"),
            SpanAttributes.LLM_RESPONSE_STOP_REASON.value: kwargs.get("stop"),
            SpanAttributes.LLM_FREQUENCY_PENALTY.value: kwargs.get("frequency_penalty"),
            SpanAttributes.LLM_PRESENCE_PENALTY.value: kwargs.get("presence_penalty"),
        }

        prompts = kwargs.get("messages")

        if prompts:
            span.add_event(
                name="gen_ai.content.prompt",
                attributes={SpanAttributes.LLM_PROMPTS.value: json.dumps(prompts)},
            )

        self.set_span_attributes(span, span_attributes)

    def set_response_attributes(self, span, response_obj):
        """
        This function is used to get span attributes for the LLM response
        """
        response_attributes = {
            "gen_ai.response_id": response_obj.get("id"),
            "gen_ai.system_fingerprint": response_obj.get("system_fingerprint"),
            SpanAttributes.LLM_RESPONSE_MODEL.value: response_obj.get("model"),
        }
        completions = []
        for choice in response_obj.get("choices", []):
            role = choice.get("message").get("role")
            content = choice.get("message").get("content")
            completions.append({"role": role, "content": content})

        span.add_event(
            name="gen_ai.content.completion",
            attributes={SpanAttributes.LLM_COMPLETIONS: json.dumps(completions)},
        )

        self.set_span_attributes(span, response_attributes)

    def set_usage_attributes(self, span, response_obj):
        """
        This function is used to get span attributes for the LLM usage
        """
        usage = response_obj.get("usage")
        if usage:
            usage_attributes = {
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS.value: usage.get(
                    "prompt_tokens"
                ),
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS.value: usage.get(
                    "completion_tokens"
                ),
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS.value: usage.get("total_tokens"),
            }
            self.set_span_attributes(span, usage_attributes)

    def set_span_attributes(self, span, attributes):
        """
        This function is used to set span attributes
        """
        for key, value in attributes.items():
            if not value:
                continue
            span.set_attribute(key, value)
