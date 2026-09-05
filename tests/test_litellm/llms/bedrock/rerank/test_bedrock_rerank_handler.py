import json
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def _rerank_with_mocked_post(model: str) -> tuple[str, dict]:
    body = {"results": [{"index": 0, "relevanceScore": 0.9}], "usage": {"search_units": 1}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=body)
    mock_response.text = json.dumps(body)
    mock_response.headers = httpx.Headers({})
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.rerank(
        model=model,
        query="capital of the United States",
        documents=["Washington, D.C.", "Carson City"],
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
    )
    post_call = client.post.call_args
    url = post_call.args[0] if post_call.args else post_call.kwargs["url"]
    return url, json.loads(post_call.kwargs["data"])


def test_region_prefixed_rerank_model_calls_that_region_with_the_bare_model_arn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    model_arn = "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"

    url, request_body = _rerank_with_mocked_post(f"bedrock/us-east-1/{model_arn}")

    assert url == "https://bedrock-agent-runtime.us-east-1.amazonaws.com/rerank"
    assert request_body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"] == {
        "modelArn": model_arn
    }
