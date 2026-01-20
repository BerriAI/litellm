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
from unittest.mock import MagicMock, Mock, patch
import pytest
import httpx

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
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_13sf6-cALnp38iZD", # streaming invocation
    ]
)
async def test_bedrock_agentcore_with_streaming(model):
    """
    Test AgentCore with streaming
    """
    print("running streming test for model=", model)
    #litellm._turn_on_debug()
    response = await litellm.acompletion(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
        messages=[
            {
                "role": "user",
                "content": "Explain machine learning in simple terms",
            }
        ],
        stream=True,
    )

    async for chunk in response:
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


def test_bedrock_agentcore_with_runtime_user_id():
    """
    Test AgentCore with runtimeUserId parameter
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
                        "content": "Hello",
                    }
                ],
                runtimeUserId="test-user-123",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify headers - user ID should be in header
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "X-Amzn-Bedrock-AgentCore-Runtime-User-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] == "test-user-123"


def test_bedrock_agentcore_with_session_and_user():
    """
    Test AgentCore with both runtimeSessionId and runtimeUserId
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
                        "content": "Test message",
                    }
                ],
                runtimeSessionId="session-abc-123",
                runtimeUserId="user-xyz-789",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify headers contain both session and user IDs
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == "session-abc-123"
        assert "X-Amzn-Bedrock-AgentCore-Runtime-User-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] == "user-xyz-789"


def test_bedrock_agentcore_with_api_key_bearer_token():
    """
    Test AgentCore with api_key parameter for JWT/Bearer token authentication
    """
    import json

    litellm._turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    test_jwt_token = "test-jwt-token-header.payload.signature"

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Test JWT authentication",
                    }
                ],
                api_key=test_jwt_token,
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify Authorization header with Bearer token
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_jwt_token}"
        assert headers["Content-Type"] == "application/json"

        # Verify the request body is JSON-encoded (not SigV4 signed)
        assert "data" in call_kwargs
        request_data = json.loads(call_kwargs["data"])
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        assert "prompt" in request_data
        assert request_data["prompt"] == "Test JWT authentication"


def test_bedrock_agentcore_with_all_parameters():
    """
    Test AgentCore with all parameters: api_key, runtimeSessionId, runtimeUserId
    """
    import json

    litellm._turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    test_jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Complete test",
                    }
                ],
                api_key=test_jwt_token,
                runtimeSessionId="full-test-session-id",
                runtimeUserId="full-test-user-id",
                qualifier="LATEST",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify URL includes qualifier
        assert "url" in call_kwargs
        url = call_kwargs["url"]
        print(f"URL: {url}")
        assert "qualifier=LATEST" in url

        # Verify all headers are present
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")

        # Check Bearer token authorization
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_jwt_token}"

        # Check session and user IDs
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == "full-test-session-id"
        assert "X-Amzn-Bedrock-AgentCore-Runtime-User-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] == "full-test-user-id"

        # Verify JSON body
        assert "data" in call_kwargs
        request_data = json.loads(call_kwargs["data"])
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        assert "prompt" in request_data
        assert request_data["prompt"] == "Complete test"


def test_bedrock_agentcore_without_api_key_uses_sigv4():
    """
    Test that AgentCore uses AWS SigV4 signing when api_key is not provided
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
                        "content": "Test SigV4",
                    }
                ],
                # No api_key provided - should use SigV4
                runtimeSessionId="sigv4-test-session",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify headers - should have AWS SigV4 headers, not Bearer token
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")

        # Should NOT have Bearer Authorization when using SigV4
        if "Authorization" in headers:
            assert not headers["Authorization"].startswith("Bearer ")
            # Should have AWS4-HMAC-SHA256 signature
            assert "AWS4-HMAC-SHA256" in headers["Authorization"]

        # Session ID should still be present
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == "sigv4-test-session"


def test_agentcore_parse_json_response():
    """
    Unit test for JSON response parsing (non-streaming)
    Verifies that content-type: application/json responses are parsed correctly
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Create a mock JSON response
    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "result": {
            "role": "assistant",
            "content": [{"text": "Hello from JSON response"}]
        }
    }

    # Parse the response
    parsed = config._get_parsed_response(mock_response)

    # Verify content extraction
    assert parsed["content"] == "Hello from JSON response"
    # JSON responses don't include usage data
    assert parsed["usage"] is None
    # Final message should be the result object
    assert parsed["final_message"] == mock_response.json.return_value["result"]


def test_agentcore_parse_sse_response():
    """
    Unit test for SSE response parsing (streaming response consumed as text)
    Verifies that text/event-stream responses are parsed correctly
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Create a mock SSE response with multiple events
    sse_data = """data: {"event":{"contentBlockDelta":{"delta":{"text":"Hello "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"from SSE"}}}}

data: {"event":{"metadata":{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}}}

data: {"message":{"role":"assistant","content":[{"text":"Hello from SSE"}]}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data

    # Parse the response
    parsed = config._get_parsed_response(mock_response)

    # Verify content extraction from final message
    assert parsed["content"] == "Hello from SSE"
    # SSE responses can include usage data
    assert parsed["usage"] is not None
    assert parsed["usage"]["inputTokens"] == 10
    assert parsed["usage"]["outputTokens"] == 5
    assert parsed["usage"]["totalTokens"] == 15
    # Final message should be present
    assert parsed["final_message"] is not None
    assert parsed["final_message"]["role"] == "assistant"


def test_agentcore_parse_sse_response_without_final_message():
    """
    Unit test for SSE response parsing when only deltas are present (no final message)
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Create a mock SSE response with only content deltas
    sse_data = """data: {"event":{"contentBlockDelta":{"delta":{"text":"First "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"second "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"third"}}}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data

    # Parse the response
    parsed = config._get_parsed_response(mock_response)

    # Content should be concatenated from deltas
    assert parsed["content"] == "First second third"
    # No final message
    assert parsed["final_message"] is None


def test_agentcore_transform_response_json():
    """
    Integration test for transform_response with JSON response
    Verifies end-to-end transformation of JSON responses to ModelResponse
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
    from litellm.types.utils import ModelResponse

    config = AmazonAgentCoreConfig()

    # Create mock JSON response
    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "result": {
            "role": "assistant",
            "content": [{"text": "Response from transform_response"}]
        }
    }
    mock_response.status_code = 200

    # Create model response
    model_response = ModelResponse()

    # Mock logging object
    mock_logging = MagicMock()

    # Transform the response
    result = config.transform_response(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging,
        request_data={},
        messages=[{"role": "user", "content": "test"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    # Verify ModelResponse structure
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Response from transform_response"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].index == 0


def test_agentcore_transform_response_sse():
    """
    Integration test for transform_response with SSE response
    Verifies end-to-end transformation of SSE responses to ModelResponse
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
    from litellm.types.utils import ModelResponse

    config = AmazonAgentCoreConfig()

    # Create mock SSE response
    sse_data = """data: {"event":{"contentBlockDelta":{"delta":{"text":"SSE "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"response"}}}}

data: {"event":{"metadata":{"usage":{"inputTokens":20,"outputTokens":10,"totalTokens":30}}}}

data: {"message":{"role":"assistant","content":[{"text":"SSE response"}]}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data
    mock_response.status_code = 200

    # Create model response
    model_response = ModelResponse()

    # Mock logging object
    mock_logging = MagicMock()

    # Transform the response
    result = config.transform_response(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging,
        request_data={},
        messages=[{"role": "user", "content": "test"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    # Verify ModelResponse structure
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "SSE response"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"

    # Verify usage data from SSE metadata
    assert hasattr(result, "usage")
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 30


def test_agentcore_synchronous_non_streaming_response():
    """
    Test that synchronous (non-streaming) AgentCore calls still work correctly
    after streaming simplification changes.

    This test verifies:
    1. Synchronous completion calls work (stream=False or no stream param)
    2. Response is properly parsed and returned as ModelResponse
    3. Content is extracted correctly
    4. Usage data is calculated when not provided by API

    This is a regression test for the streaming simplification changes
    to ensure we didn't break the non-streaming code path.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm._turn_on_debug()
    client = HTTPHandler()

    # Mock a JSON response (typical for synchronous AgentCore calls)
    mock_json_response = {
        "result": {
            "role": "assistant",
            "content": [{"text": "This is a synchronous response from AgentCore."}]
        }
    }

    # Create a mock response object
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = mock_json_response

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        # Make a synchronous (non-streaming) completion call
        response = litellm.completion(
            model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
            messages=[
                {
                    "role": "user",
                    "content": "Test synchronous response",
                }
            ],
            stream=False,  # Explicitly disable streaming
            client=client,
        )

        # Verify the response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0

        # Verify content
        message = response.choices[0].message
        assert message is not None
        assert message.content == "This is a synchronous response from AgentCore."
        assert message.role == "assistant"

        # Verify completion metadata
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].index == 0

        # Verify usage data exists (either from API or calculated)
        assert hasattr(response, "usage")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        print(f"Synchronous response: {response}")
        print(f"Content: {message.content}")
        print(f"Usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}")

