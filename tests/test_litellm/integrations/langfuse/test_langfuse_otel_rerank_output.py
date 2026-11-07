"""
Unit tests for LangfuseOtelLogger._set_observation_output.

Regression for #36537: with ``success_callback: ["langfuse_otel"]`` the
``/v1/rerank`` span reached Langfuse with Input set but Output always empty,
while the legacy ``langfuse`` callback populated it. The OTEL helper handled
``choices`` (chat completions) and ``output`` (Responses API items) but never
``RerankResponse.results``.
"""
from unittest.mock import MagicMock

from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
from litellm.types.utils import RerankResponse


def _make_span():
    span = MagicMock()
    span.set_attribute = MagicMock()
    return span


def _extract_output(span) -> str:
    """Pull the value written to OBSERVATION_OUTPUT by set_attribute."""
    for args, _ in span.set_attribute.call_args_list:
        if args[0] == LangfuseSpanAttributes.OBSERVATION_OUTPUT.value:
            return args[1]
    return None


def test_rerank_response_sets_observation_output():
    span = _make_span()
    response = RerankResponse(
        results=[
            {"index": 2, "relevance_score": 0.98},
            {"index": 0, "relevance_score": 0.41},
        ]
    )

    LangfuseOtelLogger._set_observation_output(span=span, response_obj=response)

    output = _extract_output(span)
    assert output is not None
    assert "0.98" in output
    assert "0.41" in output


def test_rerank_without_results_sets_no_output():
    span = _make_span()
    response = RerankResponse(results=None)

    LangfuseOtelLogger._set_observation_output(span=span, response_obj=response)

    assert _extract_output(span) is None


def test_none_response_is_noop():
    span = _make_span()

    LangfuseOtelLogger._set_observation_output(span=span, response_obj=None)

    span.set_attribute.assert_not_called()
