import sys
import os
import json
import time

import logfire
import litellm
import pytest
from logfire.testing import TestExporter, SimpleSpanProcessor

sys.path.insert(0, os.path.abspath("../.."))

# Testing scenarios for logfire logging:
# 1. Test logfire logging for completion
# 2. Test logfire logging for acompletion
# 3. Test logfire logging for completion while streaming is enabled
# 4. Test logfire logging for completion while streaming is enabled


@pytest.mark.skip(reason="Breaks on ci/cd")
@pytest.mark.parametrize("stream", [False, True])
def test_completion_logfire_logging(stream):
    litellm.success_callback = ["logfire"]
    litellm.set_verbose = True

    exporter = TestExporter()
    logfire.configure(
        send_to_logfire=False,
        console=False,
        processors=[SimpleSpanProcessor(exporter)],
        collect_system_metrics=False,
    )
    messages = [{"role": "user", "content": "what llm are u"}]
    temperature = 0.3
    max_tokens = 10
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    print(response)

    if stream:
        for chunk in response:
            print(chunk)

    time.sleep(5)
    exported_spans = exporter.exported_spans_as_dict()

    assert len(exported_spans) == 1
    assert (
        exported_spans[0]["attributes"]["logfire.msg"]
        == "Chat Completion with 'gpt-3.5-turbo'"
    )

    request_data = json.loads(exported_spans[0]["attributes"]["request_data"])

    assert request_data["model"] == "gpt-3.5-turbo"
    assert request_data["messages"] == messages

    assert "completion_tokens" in request_data["usage"]
    assert "prompt_tokens" in request_data["usage"]
    assert "total_tokens" in request_data["usage"]
    assert request_data["response"]["choices"][0]["message"]["content"]
    assert request_data["modelParameters"]["max_tokens"] == max_tokens
    assert request_data["modelParameters"]["temperature"] == temperature


@pytest.mark.skip(reason="Breaks on ci/cd")
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_acompletion_logfire_logging(stream):
    litellm.success_callback = ["logfire"]
    litellm.set_verbose = True

    exporter = TestExporter()
    logfire.configure(
        send_to_logfire=False,
        console=False,
        processors=[SimpleSpanProcessor(exporter)],
        collect_system_metrics=False,
    )
    messages = [{"role": "user", "content": "what llm are u"}]
    temperature = 0.3
    max_tokens = 10
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    print(response)
    if stream:
        for chunk in response:
            print(chunk)

    time.sleep(5)
    exported_spans = exporter.exported_spans_as_dict()
    print("exported_spans", exported_spans)

    assert len(exported_spans) == 1
    assert (
        exported_spans[0]["attributes"]["logfire.msg"]
        == "Chat Completion with 'gpt-3.5-turbo'"
    )

    request_data = json.loads(exported_spans[0]["attributes"]["request_data"])

    assert request_data["model"] == "gpt-3.5-turbo"
    assert request_data["messages"] == messages

    assert "completion_tokens" in request_data["usage"]
    assert "prompt_tokens" in request_data["usage"]
    assert "total_tokens" in request_data["usage"]
    assert request_data["response"]["choices"][0]["message"]["content"]
    assert request_data["modelParameters"]["max_tokens"] == max_tokens
    assert request_data["modelParameters"]["temperature"] == temperature
