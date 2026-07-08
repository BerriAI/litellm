import json
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))
import litellm


@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post")
@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_rerank_does_not_log_query_or_documents(mock_sync_post, mock_async_post, sync_mode, caplog):
    """
    rerank() used to log the full `optional_rerank_params` at INFO, which embeds the
    caller's `query` and `documents` (their raw content). That leaked request content
    into stdout regardless of `turn_off_message_logging`. The INFO line must carry the
    non-content params but never the query or documents
    """
    mock_response_data = {
        "id": "rerank-123",
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ],
        "meta": {
            "tokens": {"input_tokens": 25, "output_tokens": 0},
            "billed_units": {"total_tokens": 25},
        },
    }

    mock_response = MagicMock()
    mock_response.json = lambda: mock_response_data
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.text = json.dumps(mock_response_data)
    mock_sync_post.return_value = mock_response
    mock_async_post.return_value = mock_response

    secret_query = "PII_QUERY_ivan_petrov_account_40817810099910004312"
    secret_document = "PII_DOCUMENT_confidential_client_report_body"

    with caplog.at_level(logging.INFO, logger="LiteLLM"):
        kwargs = dict(
            model="cohere/rerank-english-v3.0",
            query=secret_query,
            documents=[secret_document, "another document"],
            top_n=2,
            custom_llm_provider="cohere",
            api_key="fake-key",
        )
        if sync_mode:
            litellm.rerank(**kwargs)
        else:
            import asyncio

            asyncio.run(litellm.arerank(**kwargs))

    param_logs = [record.getMessage() for record in caplog.records if "optional_rerank_params" in record.getMessage()]
    assert param_logs, "expected an 'optional_rerank_params' INFO log to be emitted"

    joined = "\n".join(param_logs)
    assert secret_query not in joined
    assert secret_document not in joined
    assert "top_n" in joined
