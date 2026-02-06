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

import httpx
import litellm
from unittest.mock import MagicMock, Mock, patch
import pytest

# Skip marker for integration tests that require live AWS credentials with AgentCore permissions
requires_agentcore_credentials = pytest.mark.skipif(
    os.getenv("AGENTCORE_INTEGRATION_TEST") != "true",
    reason="AgentCore integration tests require AGENTCORE_INTEGRATION_TEST=true and valid AWS credentials with bedrock-agentcore:InvokeAgentRuntime permission"
)


@requires_agentcore_credentials
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


@requires_agentcore_credentials
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


def test_agentcore_extract_reasoning_from_strands_event():
    """
    Unit test for extracting reasoning content from Strands SDK streaming events.

    Strands SDK emits reasoning events with top-level format:
    {"reasoning": true, "reasoningText": "...", "reasoning_signature": "..."}
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Test Strands SDK reasoning event format
    strands_event = {
        "reasoning": True,
        "reasoningText": "Let me think about this problem step by step...",
        "reasoning_signature": "sig123abc"
    }

    reasoning_block = config._extract_reasoning_from_event(strands_event)

    assert reasoning_block is not None
    assert "reasoningText" in reasoning_block
    assert reasoning_block["reasoningText"]["text"] == "Let me think about this problem step by step..."
    assert reasoning_block["reasoningText"]["signature"] == "sig123abc"


def test_agentcore_extract_reasoning_with_signature_alias():
    """
    Unit test for extracting reasoning with 'signature' alias (instead of reasoning_signature).
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Test with 'signature' key instead of 'reasoning_signature'
    strands_event = {
        "reasoning": True,
        "reasoningText": "Analyzing the request...",
        "signature": "alt_sig_456"
    }

    reasoning_block = config._extract_reasoning_from_event(strands_event)

    assert reasoning_block is not None
    assert reasoning_block["reasoningText"]["text"] == "Analyzing the request..."
    assert reasoning_block["reasoningText"]["signature"] == "alt_sig_456"


def test_agentcore_extract_redacted_reasoning():
    """
    Unit test for extracting redacted reasoning content.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Test redacted reasoning event
    redacted_event = {
        "reasoning": True,
        "redactedContent": "base64encodedredacteddata=="
    }

    reasoning_block = config._extract_reasoning_from_event(redacted_event)

    assert reasoning_block is not None
    assert "redactedContent" in reasoning_block
    assert reasoning_block["redactedContent"] == "base64encodedredacteddata=="


def test_agentcore_extract_reasoning_from_bedrock_converse_style():
    """
    Unit test for extracting reasoning from Bedrock Converse style nested events.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Test Bedrock Converse style reasoning event
    converse_event = {
        "event": {
            "contentBlockDelta": {
                "delta": {
                    "reasoningText": "Processing user input...",
                    "signature": "converse_sig_789"
                }
            }
        }
    }

    reasoning_block = config._extract_reasoning_from_event(converse_event)

    assert reasoning_block is not None
    assert "reasoningText" in reasoning_block
    assert reasoning_block["reasoningText"]["text"] == "Processing user input..."
    assert reasoning_block["reasoningText"]["signature"] == "converse_sig_789"


def test_agentcore_extract_reasoning_no_reasoning_event():
    """
    Unit test verifying that non-reasoning events return None.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Test regular content delta (not reasoning)
    content_event = {
        "event": {
            "contentBlockDelta": {
                "delta": {
                    "text": "Hello, this is regular content."
                }
            }
        }
    }

    reasoning_block = config._extract_reasoning_from_event(content_event)
    assert reasoning_block is None


def test_agentcore_transform_reasoning_content():
    """
    Unit test for transforming reasoning blocks to concatenated reasoning text.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    reasoning_blocks = [
        {"reasoningText": {"text": "First, I analyze the problem. ", "signature": "sig1"}},
        {"reasoningText": {"text": "Then, I consider the options. ", "signature": "sig2"}},
        {"redactedContent": "redacted_data"},  # Should be skipped
        {"reasoningText": {"text": "Finally, I reach a conclusion.", "signature": "sig3"}},
    ]

    result = config._transform_reasoning_content(reasoning_blocks)

    assert result == "First, I analyze the problem. Then, I consider the options. Finally, I reach a conclusion."


def test_agentcore_transform_thinking_blocks():
    """
    Unit test for transforming reasoning blocks to OpenAI-compatible thinking blocks.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    reasoning_blocks = [
        {"reasoningText": {"text": "Step 1: Understand the query.", "signature": "sig_step1"}},
        {"redactedContent": "some_redacted_data"},
        {"reasoningText": {"text": "Step 2: Formulate response."}},  # No signature
    ]

    thinking_blocks = config._transform_thinking_blocks(reasoning_blocks)

    assert len(thinking_blocks) == 3

    # First block - thinking with signature
    assert thinking_blocks[0]["type"] == "thinking"
    assert thinking_blocks[0]["thinking"] == "Step 1: Understand the query."
    assert thinking_blocks[0]["signature"] == "sig_step1"

    # Second block - redacted
    assert thinking_blocks[1]["type"] == "redacted_thinking"
    assert thinking_blocks[1]["data"] == "some_redacted_data"

    # Third block - thinking without signature
    assert thinking_blocks[2]["type"] == "thinking"
    assert thinking_blocks[2]["thinking"] == "Step 2: Formulate response."
    assert "signature" not in thinking_blocks[2]


def test_agentcore_parse_sse_response_with_reasoning():
    """
    Unit test for SSE response parsing with reasoning content (Strands format).
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Create SSE response with reasoning events
    sse_data = """data: {"reasoning":true,"reasoningText":"Let me analyze this...","reasoning_signature":"sig_analysis"}

data: {"reasoning":true,"reasoningText":"Now considering options...","reasoning_signature":"sig_consider"}

data: {"event":{"contentBlockDelta":{"delta":{"text":"Here is my answer."}}}}

data: {"event":{"metadata":{"usage":{"inputTokens":50,"outputTokens":100,"totalTokens":150}}}}

data: {"message":{"role":"assistant","content":[{"text":"Here is my answer."}]}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data

    parsed = config._get_parsed_response(mock_response)

    # Verify content
    assert parsed["content"] == "Here is my answer."

    # Verify usage
    assert parsed["usage"] is not None
    assert parsed["usage"]["inputTokens"] == 50
    assert parsed["usage"]["outputTokens"] == 100

    # Verify reasoning blocks were captured
    assert parsed["reasoning_content_blocks"] is not None
    assert len(parsed["reasoning_content_blocks"]) == 2
    assert parsed["reasoning_content_blocks"][0]["reasoningText"]["text"] == "Let me analyze this..."
    assert parsed["reasoning_content_blocks"][0]["reasoningText"]["signature"] == "sig_analysis"
    assert parsed["reasoning_content_blocks"][1]["reasoningText"]["text"] == "Now considering options..."


def test_agentcore_transform_response_with_reasoning():
    """
    Integration test for transform_response with reasoning content.
    Verifies that reasoning_content and thinking_blocks are populated in the response.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
    from litellm.types.utils import ModelResponse

    config = AmazonAgentCoreConfig()

    # Create mock SSE response with reasoning
    sse_data = """data: {"reasoning":true,"reasoningText":"Thinking about the problem...","reasoning_signature":"thinking_sig"}

data: {"reasoning":true,"redactedContent":"c29tZXJlZGFjdGVkZGF0YQ=="}

data: {"event":{"contentBlockDelta":{"delta":{"text":"The answer is 42."}}}}

data: {"event":{"metadata":{"usage":{"inputTokens":25,"outputTokens":10,"totalTokens":35}}}}

data: {"message":{"role":"assistant","content":[{"text":"The answer is 42."}]}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data
    mock_response.status_code = 200

    model_response = ModelResponse()
    mock_logging = MagicMock()

    result = config.transform_response(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging,
        request_data={},
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    # Verify basic response structure
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "The answer is 42."
    assert result.choices[0].message.role == "assistant"

    # Verify reasoning_content (concatenated text)
    message = result.choices[0].message
    assert hasattr(message, "reasoning_content")
    assert message.reasoning_content == "Thinking about the problem..."

    # Verify thinking_blocks (OpenAI format)
    assert hasattr(message, "thinking_blocks")
    assert len(message.thinking_blocks) == 2
    assert message.thinking_blocks[0]["type"] == "thinking"
    assert message.thinking_blocks[0]["thinking"] == "Thinking about the problem..."
    assert message.thinking_blocks[0]["signature"] == "thinking_sig"
    assert message.thinking_blocks[1]["type"] == "redacted_thinking"
    assert message.thinking_blocks[1]["data"] == "c29tZXJlZGFjdGVkZGF0YQ=="

    # Verify provider_specific_fields
    assert hasattr(message, "provider_specific_fields")
    assert "reasoningContentBlocks" in message.provider_specific_fields
    assert len(message.provider_specific_fields["reasoningContentBlocks"]) == 2


def test_agentcore_json_response_with_reasoning():
    """
    Unit test for JSON response parsing with reasoning content embedded in message.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Create mock JSON response with reasoning in content blocks
    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "result": {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "Reasoning step 1", "signature": "json_sig"}}},
                {"text": "Final answer from JSON response."}
            ]
        }
    }

    parsed = config._get_parsed_response(mock_response)

    # Verify content extraction
    assert parsed["content"] == "Final answer from JSON response."

    # Verify reasoning blocks extracted from content
    assert parsed["reasoning_content_blocks"] is not None
    assert len(parsed["reasoning_content_blocks"]) == 1
    # The reasoningContent object is added directly
    assert "reasoningText" in parsed["reasoning_content_blocks"][0]


def test_agentcore_extract_reasoning_from_agentcore_nested_format():
    """
    Unit test for extracting reasoning from AgentCore nested format.

    Our Strands agent emits reasoning via:
    {"event": {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "..."}}}}}

    This is distinct from both:
    - Strands top-level: {"reasoning": true, "reasoningText": "..."}
    - Bedrock Converse flat: {"event": {"contentBlockDelta": {"delta": {"reasoningText": "..."}}}}
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # Test AgentCore nested reasoningContent format
    agentcore_event = {
        "event": {
            "contentBlockDelta": {
                "delta": {
                    "reasoningContent": {
                        "text": "Let me analyze this step by step..."
                    }
                },
                "contentBlockIndex": 0,
            }
        }
    }

    reasoning_block = config._extract_reasoning_from_event(agentcore_event)

    assert reasoning_block is not None
    assert "reasoningText" in reasoning_block
    assert reasoning_block["reasoningText"]["text"] == "Let me analyze this step by step..."


def test_agentcore_extract_reasoning_agentcore_format_with_signature():
    """
    Unit test for AgentCore nested format with signature field.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    agentcore_event = {
        "event": {
            "contentBlockDelta": {
                "delta": {
                    "reasoningContent": {
                        "text": "Considering the options...",
                        "signature": "nested_sig_abc"
                    }
                }
            }
        }
    }

    reasoning_block = config._extract_reasoning_from_event(agentcore_event)

    assert reasoning_block is not None
    assert reasoning_block["reasoningText"]["text"] == "Considering the options..."
    assert reasoning_block["reasoningText"]["signature"] == "nested_sig_abc"

def test_agentcore_parse_sse_response_with_agentcore_reasoning_format():
    """
    Unit test for non-streaming SSE parsing with AgentCore nested reasoningContent format.
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    # SSE data with reasoningContent nested format (what our agent emits)
    sse_data = """data: {"event":{"contentBlockDelta":{"delta":{"reasoningContent":{"text":"Step 1: analyze..."}},"contentBlockIndex":0}}}

data: {"event":{"contentBlockDelta":{"delta":{"reasoningContent":{"text":"Step 2: decide..."}},"contentBlockIndex":0}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"Here is the answer."}},"contentBlockIndex":0}}

data: {"event":{"metadata":{"usage":{"inputTokens":30,"outputTokens":20,"totalTokens":50}}}}

data: {"message":{"role":"assistant","content":[{"text":"Here is the answer."}]}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data

    parsed = config._get_parsed_response(mock_response)

    assert parsed["content"] == "Here is the answer."
    assert parsed["usage"]["inputTokens"] == 30

    # Reasoning blocks should be captured from the nested format
    assert parsed["reasoning_content_blocks"] is not None
    assert len(parsed["reasoning_content_blocks"]) == 2
    assert parsed["reasoning_content_blocks"][0]["reasoningText"]["text"] == "Step 1: analyze..."
    assert parsed["reasoning_content_blocks"][1]["reasoningText"]["text"] == "Step 2: decide..."

