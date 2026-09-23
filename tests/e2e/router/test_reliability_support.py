"""Unit tests for the reliability helpers that read provider bodies, driven with
the verbatim shapes the proxy relays; nothing here needs a proxy or a provider."""

from __future__ import annotations

from typing import Final

import pytest

from e2e_http import StreamingResponse
from reliability_support import azure_prompt_filter_skipped

_SKIPPED_FILTER_BODY: Final = (
    '{"id":"chatcmpl-1","object":"chat.completion","model":"reliability-policyfail-1","choices":[{"finish_reason":'
    '"stop","index":0,"message":{"role":"assistant","content":"I can\'t comply."}}],"usage":{"prompt_tokens":64,'
    '"completion_tokens":53,"total_tokens":117},"prompt_filter_results":[{"prompt_index":0,"content_filter_results":{}}]}'
)
_RAN_FILTER_BODY: Final = (
    '{"id":"chatcmpl-2","object":"chat.completion","model":"reliability-policyfail-1","choices":[{"finish_reason":'
    '"stop","index":0,"message":{"role":"assistant","content":"Sure."}}],"usage":{"prompt_tokens":64,'
    '"completion_tokens":2,"total_tokens":66},"prompt_filter_results":[{"prompt_index":0,"content_filter_results":'
    '{"jailbreak":{"filtered":false,"detected":false},"hate":{"filtered":false,"severity":"safe"}}}]}'
)
_FALLBACK_BODY: Final = (
    '{"id":"chatcmpl-3","object":"chat.completion","model":"gpt-5.5","choices":[{"finish_reason":"stop","index":0,'
    '"message":{"role":"assistant","content":"Hello."}}],"usage":{"prompt_tokens":64,"completion_tokens":2,'
    '"total_tokens":66}}'
)
_REFUSAL_BODY: Final = (
    '{"error":{"message":"litellm.ContentPolicyViolationError: AzureException - The response was filtered",'
    '"type":"invalid_request_error","param":"prompt","code":"content_filter"}}'
)


def _response(status_code: int, body: str) -> StreamingResponse:
    return StreamingResponse(status_code=status_code, body=body)


def test_a_200_whose_prompt_filter_recorded_no_verdict_is_a_skipped_filter() -> None:
    assert azure_prompt_filter_skipped(_response(200, _SKIPPED_FILTER_BODY))


@pytest.mark.parametrize(
    ("status_code", "body"),
    (
        (200, _RAN_FILTER_BODY),
        (200, _FALLBACK_BODY),
        (400, _REFUSAL_BODY),
        (200, "not json"),
        (200, '{"prompt_filter_results":[]}'),
    ),
    ids=("filter-ran", "fallback-body", "refusal", "not-json", "empty-list"),
)
def test_every_other_body_is_not_a_skipped_filter(status_code: int, body: str) -> None:
    assert not azure_prompt_filter_skipped(_response(status_code, body))
