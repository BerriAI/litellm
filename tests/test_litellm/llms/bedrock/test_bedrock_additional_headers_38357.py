import httpx
import pytest
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.types.utils import ModelResponse

def test_bedrock_converse_populates_additional_headers():
    """
    Regression test for #38357: Bedrock Converse handler should populate
    response headers (e.g. x-amzn-RequestId) into _hidden_params['additional_headers'].
    """
    config = AmazonConverseConfig()
    
    mock_payload = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello world"}]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15
        }
    }
    
    headers = {
        "x-amzn-RequestId": "test-request-id-12345",
        "content-type": "application/json",
        "date": "Wed, 26 Aug 2026 16:00:00 GMT"
    }
    
    raw_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        headers=headers,
        request=httpx.Request("POST", "https://bedrock.test")
    )
    
    model_response = ModelResponse()
    
    res = config._transform_response(
        model="bedrock/anthropic.claude-v2",
        response=raw_response,
        model_response=model_response,
        stream=False,
        logging_obj=None,
        optional_params={},
        api_key="test-key",
        data={},
        messages=[{"role": "user", "content": "hi"}],
        encoding=None
    )
    
    additional_headers = res._hidden_params.get("additional_headers", {})
    assert "llm_provider-x-amzn-requestid" in additional_headers or "x-amzn-requestid" in additional_headers
    assert additional_headers.get("llm_provider-x-amzn-requestid") == "test-request-id-12345" or additional_headers.get("x-amzn-requestid") == "test-request-id-12345"
