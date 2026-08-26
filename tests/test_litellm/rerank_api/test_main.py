import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx


import litellm

MARKER_QUERY = "MARKER_QUERY_do_not_log_at_info"
MARKER_DOC = "MARKER_DOC_sensitive_customer_text"


def _mock_cohere_response() -> MagicMock:
    mock_response = MagicMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "meta": {
                "api_version": {"version": "1.0"},
                "billed_units": {"search_units": 1},
            },
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200
    return mock_response


def test_rerank_does_not_log_request_content_at_info(caplog):
    """Regression for #32525: rerank must not emit query/documents to logs at INFO.

    The mapped ``optional_rerank_params`` (which always contains ``query`` and
    ``documents``) bypasses ``turn_off_message_logging`` / ``redact_messages``,
    so logging it at INFO leaks raw request content into stdout and any log sink.
    """
    litellm.cohere_key = "test_api_key"
    caplog.set_level(logging.DEBUG, logger="LiteLLM")

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=_mock_cohere_response(),
    ):
        litellm.rerank(
            model="cohere/rerank-english-v3.0",
            query=MARKER_QUERY,
            documents=[MARKER_DOC, "unrelated"],
            top_n=2,
        )

    litellm_records = [r for r in caplog.records if r.name == "LiteLLM"]

    info_or_above = [
        r.getMessage()
        for r in litellm_records
        if r.levelno >= logging.INFO and (MARKER_QUERY in r.getMessage() or MARKER_DOC in r.getMessage())
    ]
    assert not info_or_above, f"rerank leaked request content at INFO+: {info_or_above}"

    optional_params_logs = [r for r in litellm_records if "optional_rerank_params" in r.getMessage()]
    assert optional_params_logs, "expected the optional_rerank_params line to be logged"
    assert all(
        r.levelno == logging.DEBUG for r in optional_params_logs
    ), "optional_rerank_params must be logged at DEBUG, not INFO"


TOGETHER_RERANK_BODY = {
    "id": "rerank-mock-id",
    "results": [{"index": 0, "relevance_score": 0.95}],
    "usage": {"prompt_tokens": 10, "total_tokens": 10},
}


def test_together_rerank_defaults_to_together_ai_host(respx_mock: respx.MockRouter, monkeypatch):
    """Regression for the Together host migration: rerank used to hardcode
    https://api.together.xyz/v1/rerank. The default must now be api.together.ai."""
    monkeypatch.delenv("TOGETHER_AI_API_BASE", raising=False)

    mock_route = respx_mock.post("https://api.together.ai/v1/rerank")
    mock_route.return_value = httpx.Response(200, json=TOGETHER_RERANK_BODY)

    response = litellm.rerank(
        model="together_ai/mixedbread-ai/mxbai-rerank-large-v2",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-together-key",
    )

    assert mock_route.called
    assert response.results[0]["relevance_score"] == 0.95


def test_together_rerank_honors_api_base(respx_mock: respx.MockRouter):
    """Regression: a custom api_base was silently ignored by the Together rerank handler."""
    mock_route = respx_mock.post("https://custom-together.example/v1/rerank")
    mock_route.return_value = httpx.Response(200, json=TOGETHER_RERANK_BODY)

    litellm.rerank(
        model="together_ai/mixedbread-ai/mxbai-rerank-large-v2",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-together-key",
        api_base="https://custom-together.example/v1",
    )

    assert mock_route.called
    assert mock_route.calls[0].request.headers["authorization"] == "Bearer fake-together-key"


@pytest.mark.asyncio
async def test_together_rerank_async_honors_env_api_base(respx_mock: respx.MockRouter, monkeypatch):
    """Regression: TOGETHER_AI_API_BASE was honored by chat but ignored by rerank."""
    monkeypatch.setenv("TOGETHER_AI_API_BASE", "https://env-together.example/v1")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")

    mock_route = respx_mock.post("https://env-together.example/v1/rerank")
    mock_route.return_value = httpx.Response(200, json=TOGETHER_RERANK_BODY)

    response = await litellm.arerank(
        model="together_ai/mixedbread-ai/mxbai-rerank-large-v2",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-together-key",
    )

    assert mock_route.called
    assert response.results[0]["relevance_score"] == 0.95
