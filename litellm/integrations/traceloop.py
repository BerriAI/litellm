import traceback

from litellm._logging import verbose_logger


class TraceloopLogger:
    """
    WARNING: DEPRECATED
    Use the OpenTelemetry standard integration instead
    """

    def __init__(self):
        try:
            from traceloop.sdk import Traceloop
            from traceloop.sdk.tracing.tracing import TracerWrapper
        except ModuleNotFoundError as e:
            verbose_logger.error(
                f"Traceloop not installed, try running 'pip install traceloop-sdk' to fix this error: {e}\n{traceback.format_exc()}"
            )
            raise e

        Traceloop.init(
            app_name="Litellm-Server",
            disable_batch=True,
        )
        self.tracer_wrapper = TracerWrapper()

    def log_event(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
        user_id,
        print_verbose,
        level="DEFAULT",
        status_message=None,
    ):
        from opentelemetry.semconv.ai import SpanAttributes
        from opentelemetry.trace import SpanKind, Status, StatusCode

        try:
            print_verbose(
                f"Traceloop Logging - Enters logging function for model {kwargs}"
            )

            tracer = self.tracer_wrapper.get_tracer()

            optional_params = kwargs.get("optional_params", {})
            start_time = int(start_time.timestamp())
            end_time = int(end_time.timestamp())
            span = tracer.start_span(
                "litellm.completion", kind=SpanKind.CLIENT, start_time=start_time
            )

            if span.is_recording():
                span.set_attribute(
                    SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model")
                )
                if "stop" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_CHAT_STOP_SEQUENCES,
                        optional_params.get("stop"),
                    )
                if "frequency_penalty" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_FREQUENCY_PENALTY,
                        optional_params.get("frequency_penalty"),
                    )
                if "presence_penalty" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_PRESENCE_PENALTY,
                        optional_params.get("presence_penalty"),
                    )
                if "top_p" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_TOP_P, optional_params.get("top_p")
                    )
                if "tools" in optional_params or "functions" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_FUNCTIONS,
                        optional_params.get("tools", optional_params.get("functions")),
                    )
                if "user" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_USER, optional_params.get("user")
                    )
                if "max_tokens" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_MAX_TOKENS,
                        kwargs.get("max_tokens"),
                    )
                if "temperature" in optional_params:
                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_TEMPERATURE,  # type: ignore
                        kwargs.get("temperature"),
                    )

                for idx, prompt in enumerate(kwargs.get("messages")):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{idx}.role",
                        prompt.get("role"),
                    )
                    span.set_attribute(
                        f"{SpanAttributes.LLM_PROMPTS}.{idx}.content",
                        prompt.get("content"),
                    )

                span.set_attribute(
                    SpanAttributes.LLM_RESPONSE_MODEL, response_obj.get("model")
                )
                usage = response_obj.get("usage")
                if usage:
                    span.set_attribute(
                        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                        usage.get("total_tokens"),
                    )
                    span.set_attribute(
                        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                        usage.get("completion_tokens"),
                    )
                    span.set_attribute(
                        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                        usage.get("prompt_tokens"),
                    )

                for idx, choice in enumerate(response_obj.get("choices")):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.finish_reason",
                        choice.get("finish_reason"),
                    )
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role",
                        choice.get("message").get("role"),
                    )
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content",
                        choice.get("message").get("content"),
                    )

            if (
                level == "ERROR"
                and status_message is not None
                and isinstance(status_message, str)
            ):
                span.record_exception(Exception(status_message))
                span.set_status(Status(StatusCode.ERROR, status_message))

            span.end(end_time)

        except Exception as e:
            print_verbose(f"Traceloop Layer Error - {e}")
