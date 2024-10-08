import asyncio
import logging
import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig

verbose_logger.setLevel(logging.DEBUG)


class TestOpenTelemetry(OpenTelemetry):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in async_log_success_event for TestOpenTelemetry kwargs=", kwargs)
        self.kwargs = kwargs
        await super().async_log_success_event(
            kwargs, response_obj, start_time, end_time
        )


@pytest.mark.asyncio
async def test_awesome_otel_with_message_logging_off():
    litellm.set_verbose = True

    otel_logger = TestOpenTelemetry(
        message_logging=False, config=OpenTelemetryConfig(exporter="console")
    )

    litellm.callbacks = [otel_logger]
    litellm.success_callback = []
    litellm.failure_callback = []

    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hi",
    )
    print("response", response)

    await asyncio.sleep(5)

    assert otel_logger.kwargs["messages"] == [
        {"role": "user", "content": "redacted-by-litellm"}
    ]


@pytest.mark.asyncio
@pytest.mark.skip(reason="Local only test. WIP.")
async def test_async_otel_callback():
    exporter = InMemorySpanExporter()
    litellm.set_verbose = True
    litellm.callbacks = [OpenTelemetry(OpenTelemetryConfig(exporter=exporter))]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
        user="OTEL_USER",
    )

    await asyncio.sleep(4)

    spans = exporter.get_finished_spans()
    print("spans", spans)
    assert len(spans) == 2


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-3-opus-20240229"],
)
@pytest.mark.skip(reason="Local only test. WIP.")
def test_completion_claude_3_function_call_with_otel(model):
    litellm.set_verbose = True

    litellm.callbacks = [OpenTelemetry(OpenTelemetryConfig())]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            drop_params=True,
        )

        print("response from LiteLLM", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
