import os
import sys
import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from langtrace_python_sdk import langtrace

import litellm

sys.path.insert(0, os.path.abspath("../.."))


@pytest.fixture()
def exporter():
    exporter = InMemorySpanExporter()
    langtrace.init(batch=False, custom_remote_exporter=exporter)
    litellm.success_callback = ["langtrace"]
    litellm.set_verbose = True

    return exporter


@pytest.mark.parametrize("model", ["claude-2.1", "gpt-3.5-turbo"])
def test_langtrace_logging(exporter, model):
    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "This is a test"}],
        max_tokens=1000,
        temperature=0.7,
        timeout=5,
        mock_response="hi",
    )
