"""
Test Bedrock AgentCore integration
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)

import litellm
from unittest.mock import MagicMock, patch
import pytest

import pytest

@pytest.mark.parametrize(
    "model", [
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_13sf6-cALnp38iZD", # non-streaming invocation
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC", # streaming invocation
    ]
)
def test_bedrock_agentcore_basic(model):
    """
    Test AgentCore invocation parameterized by model
    """
    litellm._turn_on_debug()
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Explain machine learning in simple terms"}],
    )
    print("response from agentcore=", response.model_dump_json(indent=4))
    # Assert that the message content has a response with some length
    assert response.choices[0].message.content
    assert len(response.choices[0].message.content) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", [
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/non_stream_agent-mdfwS2DlAu", # non-streaming invocation
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC", # streaming invocation
    ]
)
async def test_bedrock_agentcore_with_streaming(model):
    """
    Test AgentCore with streaming
    """
    #litellm._turn_on_debug()
    response = litellm.completion(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
        messages=[
            {
                "role": "user",
                "content": "Explain machine learning in simple terms",
            }
        ],
        stream=True,
    )

    for chunk in response:
        print("chunk=", chunk)


def test_bedrock_agentcore_with_custom_params():
    """
    Test AgentCore request structure with custom parameters
    """
    import json
    
    litellm._turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Explain machine learning in simple terms",
                    }
                ],
                runtimeSessionId="litellm-test-session-id-12345678901234567890",
                qualifier="DEFAULT",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")
        
        # Verify URL structure - should include ARN and qualifier
        assert "url" in call_kwargs
        url = call_kwargs["url"]
        print(f"URL: {url}")
        assert "/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aus-west-2%3A888602223428%3Aruntime%2Fhosted_agent_r9jvp-3ySZuRHjLC/invocations" in url
        assert "qualifier=DEFAULT" in url
        
        # Verify headers - session ID should be in header
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == "litellm-test-session-id-12345678901234567890"
        
        # Verify the request body - should just be the payload
        assert "data" in call_kwargs or "json" in call_kwargs
        
        # Parse the request data
        if "data" in call_kwargs:
            request_data = json.loads(call_kwargs["data"])
        else:
            request_data = call_kwargs["json"]
        
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        
        # Body should just contain the prompt
        assert "prompt" in request_data
        assert request_data["prompt"] == "Explain machine learning in simple terms"

