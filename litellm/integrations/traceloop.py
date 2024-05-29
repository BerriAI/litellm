class TraceloopLogger:
    def __init__(self):
        from traceloop.sdk.tracing.tracing import TracerWrapper
        from traceloop.sdk import Traceloop

        Traceloop.init(app_name="Litellm-Server", disable_batch=True)
        self.tracer_wrapper = TracerWrapper()

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        from opentelemetry.trace import SpanKind
        from opentelemetry.semconv.ai import SpanAttributes

        try:
            tracer = self.tracer_wrapper.get_tracer()

            model = kwargs.get("model")

            # LiteLLM uses the standard OpenAI library, so it's already handled by Traceloop SDK
            if kwargs.get("litellm_params").get("custom_llm_provider") == "openai":
                return

            optional_params = kwargs.get("optional_params", {})
            with tracer.start_as_current_span(
                "litellm.completion",
                kind=SpanKind.CLIENT,
            ) as span:
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
                            SpanAttributes.LLM_TOP_P, optional_params.get("top_p")
                        )
                    if "tools" in optional_params or "functions" in optional_params:
                        span.set_attribute(
                            SpanAttributes.LLM_REQUEST_FUNCTIONS,
                            optional_params.get(
                                "tools", optional_params.get("functions")
                            ),
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
