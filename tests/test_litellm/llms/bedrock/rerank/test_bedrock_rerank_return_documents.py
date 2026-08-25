import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import litellm
from litellm.llms.bedrock.base_aws_llm import Boto3CredentialsInfo
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

MODEL = "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
QUERY = "What is the capital of the United States?"
DOCUMENTS = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Northern Mariana Islands capital is Saipan.",
    "Washington, D.C. is the capital of the United States.",
]
BEDROCK_RESPONSE = {
    "results": [
        {"index": 2, "relevanceScore": 0.95},
        {"index": 0, "relevanceScore": 0.10},
        {"index": 1, "relevanceScore": 0.05},
    ],
    "usage": {"search_units": 1},
}


def _mock_credentials() -> Boto3CredentialsInfo:
    creds = MagicMock()
    creds.access_key = "test-access-key"
    creds.secret_key = "test-secret-key"
    creds.token = None
    return Boto3CredentialsInfo(
        credentials=creds,
        aws_region_name="us-west-2",
        aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-west-2.amazonaws.com",
    )


def test_bedrock_rerank_return_documents_true_populates_document_text():
    client = HTTPHandler()
    with (
        patch.object(client, "post") as mock_post,
        patch(  # test-quality-ok: same SigV4 credential mock as test_bedrock_rerank_header_forwarding.py; AWS signing requires real infrastructure
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=_mock_credentials(),
        ),
        patch("botocore.auth.SigV4Auth", return_value=MagicMock()),
    ):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(BEDROCK_RESPONSE)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        response = litellm.rerank(
            model=MODEL,
            query=QUERY,
            documents=DOCUMENTS,
            top_n=3,
            return_documents=True,
            client=client,
            aws_region_name="us-west-2",
        )

    assert response.results is not None
    for result in response.results:
        assert "document" in result, f"return_documents=True must populate document, got {result}"
        assert result["document"]["text"] == DOCUMENTS[result["index"]]


def test_bedrock_rerank_return_documents_false_omits_document_field():
    client = HTTPHandler()
    with (
        patch.object(client, "post") as mock_post,
        patch(  # test-quality-ok: same SigV4 credential mock as test_bedrock_rerank_header_forwarding.py; AWS signing requires real infrastructure
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=_mock_credentials(),
        ),
        patch("botocore.auth.SigV4Auth", return_value=MagicMock()),
    ):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(BEDROCK_RESPONSE)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        response = litellm.rerank(
            model=MODEL,
            query=QUERY,
            documents=DOCUMENTS,
            top_n=3,
            return_documents=False,
            client=client,
            aws_region_name="us-west-2",
        )

    assert response.results is not None
    for result in response.results:
        assert "document" not in result, f"return_documents=False must omit document, got {result}"


@pytest.mark.asyncio
async def test_bedrock_rerank_return_documents_true_async():
    client = AsyncHTTPHandler()
    with (
        patch.object(client, "post", new_callable=AsyncMock) as mock_post,
        patch(  # test-quality-ok: same SigV4 credential mock as test_bedrock_rerank_header_forwarding.py; AWS signing requires real infrastructure
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=_mock_credentials(),
        ),
        patch("botocore.auth.SigV4Auth", return_value=MagicMock()),
    ):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(BEDROCK_RESPONSE)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        response = await litellm.arerank(
            model=MODEL,
            query=QUERY,
            documents=DOCUMENTS,
            top_n=3,
            return_documents=True,
            client=client,
            aws_region_name="us-west-2",
        )

    assert response.results is not None
    for result in response.results:
        assert result["document"]["text"] == DOCUMENTS[result["index"]]


@pytest.mark.asyncio
async def test_bedrock_rerank_return_documents_false_async():
    client = AsyncHTTPHandler()
    with (
        patch.object(client, "post", new_callable=AsyncMock) as mock_post,
        patch(  # test-quality-ok: same SigV4 credential mock as test_bedrock_rerank_header_forwarding.py; AWS signing requires real infrastructure
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=_mock_credentials(),
        ),
        patch("botocore.auth.SigV4Auth", return_value=MagicMock()),
    ):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(BEDROCK_RESPONSE)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        response = await litellm.arerank(
            model=MODEL,
            query=QUERY,
            documents=DOCUMENTS,
            top_n=3,
            return_documents=False,
            client=client,
            aws_region_name="us-west-2",
        )

    assert response.results is not None
    for result in response.results:
        assert "document" not in result, f"return_documents=False must omit document, got {result}"


def test_bedrock_rerank_dict_document_without_text_key_skips_back_fill():
    from litellm.llms.bedrock.rerank.transformation import BedrockRerankConfig

    json_documents = [
        {"title": "zero", "body": "json zero body"},
        {"title": "one", "body": "json one body"},
        {"title": "two", "body": "json two body"},
    ]
    response = BedrockRerankConfig()._transform_response(
        BEDROCK_RESPONSE, documents=json_documents, return_documents=True
    )
    assert response.results is not None
    for result in response.results:
        assert "document" not in result, f"dict documents without 'text' key have no back-fill target, got {result}"


def test_bedrock_rerank_non_integer_index_skips_back_fill():
    from litellm.llms.bedrock.rerank.transformation import BedrockRerankConfig

    response = BedrockRerankConfig()._transform_response(
        {"results": [{"index": "2", "relevanceScore": 0.9}]},
        documents=DOCUMENTS,
        return_documents=True,
    )
    assert response.results is not None
    assert len(response.results) == 1
    assert "document" not in response.results[0]


def test_bedrock_rerank_out_of_range_index_skips_back_fill():
    from litellm.llms.bedrock.rerank.transformation import BedrockRerankConfig

    response = BedrockRerankConfig()._transform_response(
        {"results": [{"index": 99, "relevanceScore": 0.9}]},
        documents=DOCUMENTS,
        return_documents=True,
    )
    assert response.results is not None
    assert len(response.results) == 1
    assert "document" not in response.results[0]
