"""Shared fixtures for the reasoning_effort grid e2e suite.

VCR wiring (Redis-backed cassette persister, auto-application of
``@pytest.mark.vcr`` to every collected item, cassette-cache health summary)
is inherited from ``tests/llm_translation/conftest.py``. This file only
contributes the ``wire_capture`` fixture, which records the wire body
LiteLLM sends upstream so each cell can inspect it.
"""

from typing import Any, Dict, List, Optional

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger


class _WireBodyCapture(CustomLogger):
    """Pre-call hook that records the outgoing wire body LiteLLM sends upstream.

    ``complete_input_dict`` is the fully transformed provider request as set
    by every provider transformation in ``litellm/llms/**``. Capturing it here
    means a regression anywhere in the transformation chain (strip, rename,
    drop) surfaces as an assertion failure on the cell that depends on it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: List[Dict[str, Any]] = []

    def log_pre_api_call(self, model, messages, kwargs):
        self.records.append(
            {
                "model": model,
                "body": kwargs.get("additional_args", {}).get("complete_input_dict"),
                "api_base": kwargs.get("additional_args", {}).get("api_base"),
            }
        )

    async def async_log_pre_api_call(self, model, messages, kwargs):
        self.log_pre_api_call(model, messages, kwargs)

    def latest(self) -> Optional[Dict[str, Any]]:
        return self.records[-1] if self.records else None


@pytest.fixture()
def wire_capture():
    capture = _WireBodyCapture()
    previous = list(litellm.callbacks)
    litellm.callbacks = previous + [capture]
    try:
        yield capture
    finally:
        litellm.callbacks = previous
