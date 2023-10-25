class TraceloopLogger:
    def __init__(self):
        from traceloop.sdk.tracing.tracing import TracerWrapper

        self.tracer_wrapper = TracerWrapper()

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        from opentelemetry.trace import SpanKind
        from opentelemetry.semconv.ai import SpanAttributes

        try:
            tracer = self.tracer_wrapper.get_tracer()

            model = kwargs.get("model")

            # LiteLLM uses the standard OpenAI library, so it's already handled by Traceloop SDK
            if "gpt" in model:
                return

            with tracer.start_as_current_span(
                "litellm.completion",
                kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model")
                    )
                    span.set_attribute(
                        SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens")
                    )
                    span.set_attribute(
                        SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature")
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

        except Exception as e:
            print_verbose(f"Traceloop Layer Error - {e}")
