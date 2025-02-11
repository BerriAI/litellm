import os
import sys
import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm

sys.path.insert(0, os.path.abspath("../.."))


@pytest.fixture()
@pytest.mark.skip(reason="Traceloop use `otel` integration instead")
def exporter():
    from traceloop.sdk import Traceloop

    exporter = InMemorySpanExporter()
    Traceloop.init(
        app_name="test_litellm",
        disable_batch=True,
        exporter=exporter,
    )
    litellm.success_callback = ["traceloop"]
    litellm.set_verbose = True

    return exporter


@pytest.mark.skip(reason="moved to using 'otel' for logging")
@pytest.mark.parametrize("model", ["claude-3-5-haiku-20241022", "gpt-3.5-turbo"])
@pytest.mark.skip(reason="Traceloop use `otel` integration instead")
def test_traceloop_logging(exporter, model):
    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "This is a test"}],
        max_tokens=1000,
        temperature=0.7,
        timeout=5,
        mock_response="hi",
    )
