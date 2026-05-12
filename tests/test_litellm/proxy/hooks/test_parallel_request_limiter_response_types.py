"""
Regression tests for response-type coverage in parallel_request_limiter TPM
counting. Guards against #27738 — EmbeddingResponse /
TextCompletionResponse / ResponsesAPIResponse were not counted toward
per-key / per-team / per-org TPM because only `isinstance(response_obj,
ModelResponse)` was checked.
"""

import pytest
from unittest.mock import MagicMock

from litellm import (
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
)
from litellm.proxy.hooks.parallel_request_limiter import _extract_total_tokens

try:
    from litellm.types.llms.openai import ResponsesAPIResponse
    _HAS_RESPONSES_API = True
except ImportError:
    ResponsesAPIResponse = None
    _HAS_RESPONSES_API = False


def test_extract_total_tokens_model_response():
    r = ModelResponse()
    r.usage = MagicMock(total_tokens=123)
    assert _extract_total_tokens(r) == 123


def test_extract_total_tokens_embedding_response():
    r = EmbeddingResponse()
    r.usage = MagicMock(total_tokens=456)
    assert _extract_total_tokens(r) == 456


def test_extract_total_tokens_text_completion_response():
    r = TextCompletionResponse()
    r.usage = MagicMock(total_tokens=789)
    assert _extract_total_tokens(r) == 789


@pytest.mark.skipif(not _HAS_RESPONSES_API, reason="ResponsesAPIResponse not available in this version")
def test_extract_total_tokens_responses_api_response():
    # ResponsesAPIResponse has required Pydantic fields (id, created_at, output)
    # so we use MagicMock with spec to satisfy isinstance checks without
    # triggering validation errors.
    r = MagicMock(spec=ResponsesAPIResponse)
    r.usage = MagicMock(total_tokens=321)
    assert _extract_total_tokens(r) == 321


def test_extract_total_tokens_unsupported_type_returns_zero():
    """Silent no-op for unknown response types — matches prior behavior."""
    class UnknownResponse:
        pass
    r = UnknownResponse()
    r.usage = MagicMock(total_tokens=999)
    assert _extract_total_tokens(r) == 0


def test_extract_total_tokens_none_usage_returns_zero():
    r = ModelResponse()
    r.usage = None
    assert _extract_total_tokens(r) == 0
