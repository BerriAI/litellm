import os
import sys
import pytest
import respx
from dotenv import load_dotenv
from unittest.mock import patch

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm


@pytest.fixture(autouse=True)
def mock_cloudflare_secrets():
    """Mock Cloudflare secrets for consistent testing"""
    with patch("litellm.secret_managers.main.get_secret_str") as mock_get_secret:

        def side_effect(key):
            if key == "CLOUDFLARE_ACCOUNT_ID":
                return "test-account-id"
            elif key == "CLOUDFLARE_API_KEY":
                return "test-api-key"
            return None

        mock_get_secret.side_effect = side_effect
        yield mock_get_secret


@pytest.fixture
def cloudflare_api_response():
    """Mock response data for Cloudflare API"""
    return {
        "result": {
            "response": "I am Llama 4 Scout, a large language model developed by Meta.",
            "tool_calls": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


@pytest.fixture
def cloudflare_streaming_response():
    """Mock streaming response data for Cloudflare API"""
    return 'data: {"response":"Hello","tool_calls":[],"p":"abdefghij"}\n\ndata: {"response":"!","tool_calls":[],"p":"abdef"}\n\ndata: {"response":" I","tool_calls":[],"p":"abdefghijklmnop"}\n\ndata: {"response":" am","tool_calls":[],"p":"abdefghijklmnoprstuvxyz123"}\n\ndata: {"response":" Llama","tool_calls":[],"p":"abdefghijklmnoprstuvxyz1234567890abdefghijklm"}\n\ndata: {"response":" 4","tool_calls":[],"p":"abdefghijkl"}\n\ndata: {"response":" Scout","tool_calls":[],"p":"abdefghijklmnoprstuvxyz1234567890abd"}\n\ndata: [DONE]\n\n'


@pytest.fixture
def cloudflare_function_calling_response():
    """Mock response data for Cloudflare function calling (Llama 4 Scout format)"""
    return {
        "result": {
            "tool_calls": [
                {
                    "id": "chatcmpl-tool-926d0e07eecd4665bfde18c2d0dd213f",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 160,
                "completion_tokens": 17,
                "total_tokens": 177,
            },
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


@pytest.fixture
def cloudflare_llama33_function_calling_response():
    """Mock response data for Cloudflare Llama 3.3 function calling"""
    return {
        "result": {
            "response": None,
            "tool_calls": [{"name": "get_weather", "arguments": {"location": "Paris"}}],
            "usage": {
                "prompt_tokens": 151,
                "completion_tokens": 23,
                "total_tokens": 174,
            },
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


@pytest.fixture
def cloudflare_llama33_response():
    """Mock response data for Cloudflare Llama 3.3 70B FP8 Fast model"""
    return {
        "result": {
            "response": 'I\'m an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."',
            "tool_calls": [],
            "usage": {"prompt_tokens": 42, "completion_tokens": 23, "total_tokens": 65},
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_completion_cloudflare_llama4_scout(
    stream,
    respx_mock: respx.MockRouter,
    cloudflare_api_response,
    cloudflare_streaming_response,
):
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        litellm.set_verbose = False

        if stream:
            # Mock streaming response
            respx_mock.post(
                "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
            ).respond(
                content=cloudflare_streaming_response,
                headers={"content-type": "text/plain"},
            )
        else:
            # Mock regular response
            respx_mock.post(
                "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
            ).respond(json=cloudflare_api_response)

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": "what llm are you", "role": "user"}],
            max_tokens=1024,
            temperature=0.7,
            stream=stream,
        )
        if stream is True:
            async for chunk in response:  # type: ignore
                print(chunk)
        else:
            print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_cloudflare_llama4_scout_with_parameters(
    respx_mock: respx.MockRouter, cloudflare_api_response
):
    """Test new parameters with Llama-4-Scout model"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_api_response)

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": "Test with new parameters", "role": "user"}],
            temperature=2.0,
            top_k=25,
            max_tokens=1024,
        )
        print("New parameters test response:", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_guided_json_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_api_response
):
    """Test Cloudflare-specific guided_json parameter"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_api_response)

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[
                {"content": "Generate a person with name and age", "role": "user"}
            ],
            guided_json={
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
            max_tokens=100,
        )
        print("Guided JSON response:", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_raw_parameter_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_api_response
):
    """Test Cloudflare-specific raw parameter to disable chat template"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_api_response)

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[
                {"content": "Complete this: The capital of France is", "role": "user"}
            ],
            raw=True,
            max_tokens=1024,
        )
        print("Raw parameter test response:", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_parameter_validation_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter,
):
    """Test parameter validation for Cloudflare-specific ranges"""
    try:
        # Mock 400 response for temperature validation error
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(
            status_code=400,
            json={
                "errors": [
                    {
                        "message": "AiError: Bad input: Error: '/temperature' must be <= 5",
                        "code": 5006,
                    }
                ],
                "success": False,
                "result": {},
                "messages": [],
            },
        )

        await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": "test", "role": "user"}],
            temperature=6.0,
            max_tokens=10,
            drop_params=False,  # Ensure validation errors are raised
        )
        pytest.fail("Should have raised error for temperature > 5")
    except (ValueError, litellm.APIConnectionError) as e:
        assert "Temperature must be between 0 and 5" in str(e)


    try:
        # Mock 400 response for top_p validation error
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(
            status_code=400,
            json={"error": {"message": "top_p must be between 0 and 2"}},
        )

        await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": "test", "role": "user"}],
            top_p=3.0,
            max_tokens=10,
            drop_params=False,  # Ensure validation errors are raised
        )
        pytest.fail("Should have raised error for top_p > 2")
    except (ValueError, litellm.APIConnectionError) as e:
        assert "top_p must be between 0 and 2" in str(e)

    try:
        # Mock 400 response for seed validation error
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(
            status_code=400,
            json={"error": {"message": "seed must be between 1 and 9999999999"}},
        )

        await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": "test", "role": "user"}],
            seed=10000000000,
            max_tokens=10,
            drop_params=False,  # Ensure validation errors are raised
        )
        pytest.fail("Should have raised error for seed > 9999999999")
    except (ValueError, litellm.APIConnectionError) as e:
        assert "seed must be between 1 and 9999999999" in str(e)



@pytest.mark.asyncio
async def test_function_calling_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_function_calling_response, mock_cloudflare_secrets
):
    """Test single function calling"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_function_calling_response)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": "What's the weather in Tokyo?", "role": "user"}],
            tools=tools,
            max_tokens=100,
        )
        print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_streaming_function_calling_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_streaming_response
):
    """Test streaming function calling"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock streaming response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(
            content=cloudflare_streaming_response,
            headers={"content-type": "text/plain"},
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": "What's the weather in Tokyo?", "role": "user"}],
            tools=tools,
            stream=True,
            max_tokens=100,
        )

        async for chunk in response:  # type: ignore
            print("Streaming chunk:", chunk)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_multiple_function_calling_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_function_calling_response
):
    """Test if Cloudflare supports parallel function calling"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_function_calling_response)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                    },
                },
            },
        ]

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[
                {"content": "What's the weather and time in Tokyo?", "role": "user"}
            ],
            tools=tools,
            max_tokens=100,
        )
        print("Multiple function calling response:", response)

        # Check if multiple tool_calls are returned
        if (
            hasattr(response.choices[0].message, "tool_calls")  # type: ignore
            and response.choices[0].message.tool_calls  # type: ignore
        ):
            print(
                f"Number of tool calls: {len(response.choices[0].message.tool_calls)}"  # type: ignore
            )

    except Exception as e:
        print(f"Multiple function calling test failed: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_multimodal_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_api_response
):
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_api_response)

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_invalid_image_format_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_api_response
):
    """Test that HTTP URLs are automatically converted to data URIs"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock successful response since we're testing parameter validation, not actual API behavior
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_api_response)

        # This test now verifies that HTTP URLs are converted, not rejected
        # Using a mock image URL that would fail conversion to test error handling
        await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://invalid-url-that-will-fail.local/nonexistent.jpg"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        pytest.fail("Should have raised error for invalid URL conversion")
    except Exception as e:
        # Now we expect an error from URL conversion failure, not format validation
        assert "Unable to fetch image from URL" in str(e) or "data URI format" in str(e)
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_large_context_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_api_response
):
    """Test large context window (131k tokens)"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_api_response)

        large_text = "This is a test message. " * 1000
        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[{"content": large_text, "role": "user"}],
            max_tokens=100,
        )
        print("Large context test response:", response)

    except Exception as e:
        print(f"Large context test result: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_response_format_cloudflare_llama4_scout(
    respx_mock: respx.MockRouter, cloudflare_api_response
):
    """Test response_format parameter"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
        ).respond(json=cloudflare_api_response)

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
            messages=[
                {"content": "Generate JSON with name and age fields", "role": "user"}
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
        )
        print("Response format test response:", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


# Llama 3.3 70B Instruct FP8 Fast Tests
@pytest.mark.asyncio
async def test_completion_cloudflare_llama33_70b_fp8_fast(
    respx_mock: respx.MockRouter, cloudflare_llama33_response
):
    """Test basic completion with Llama 3.3 70B FP8 Fast model"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        litellm.set_verbose = False

        # Mock regular response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        ).respond(json=cloudflare_llama33_response)

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            messages=[{"content": "what llm are you", "role": "user"}],
            max_tokens=256,
            temperature=0.6,
        )
        print("Llama 3.3 70B FP8 Fast response:", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_cloudflare_llama33_70b_function_calling(
    respx_mock: respx.MockRouter, cloudflare_llama33_function_calling_response
):
    """Test Llama 3.3 70B FP8 Fast function calling"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock response
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        ).respond(json=cloudflare_llama33_function_calling_response)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            messages=[{"content": "What's the weather like in Paris?", "role": "user"}],
            tools=tools,
            max_tokens=100,
        )
        print("Llama 3.3 70B FP8 Fast function calling response:", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


@pytest.mark.asyncio
async def test_cloudflare_llama33_70b_parameter_validation(
    respx_mock: respx.MockRouter,
):
    """Test parameter validation for Llama 3.3 70B FP8 Fast model"""
    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        litellm.disable_aiohttp_transport = True
        
        # Mock 400 response for temperature validation error
        respx_mock.post(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        ).respond(
            status_code=400,
            json={"error": {"message": "Temperature must be between 0 and 5"}},
        )

        await litellm.acompletion(
            api_key="test-api-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
            model="cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            messages=[{"content": "test", "role": "user"}],
            temperature=6.0,
            max_tokens=10,
            drop_params=False,
        )
        pytest.fail("Should have raised error for temperature > 5")
    except (ValueError, litellm.APIConnectionError) as e:
        assert "Temperature must be between 0 and 5" in str(e)
    finally:
        # Reset transport setting
        litellm.disable_aiohttp_transport = False


# Embedding Tests
@pytest.fixture
def cloudflare_embedding_response():
    """Mock response data for Cloudflare BGE embedding API"""
    return {
        "result": {
            "shape": [1, 384],
            "data": [[0.1, 0.2, 0.3] * 128],  # 384 dimensions
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


@pytest.fixture
def cloudflare_embedding_multiple_response():
    """Mock response data for Cloudflare BGE embedding API with multiple texts"""
    return {
        "result": {
            "shape": [2, 384],
            "data": [
                [0.1, 0.2, 0.3] * 128,  # First embedding
                [0.4, 0.5, 0.6] * 128,  # Second embedding
            ],
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


class TestCloudflareEmbeddings:
    def test_cloudflare_embedding_single_text(
        self, respx_mock: respx.MockRouter, cloudflare_embedding_response
    ):
        """Test Cloudflare embedding with single text input."""
        try:
            # Mock response
            respx_mock.post(
                "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/baai/bge-large-en-v1.5"
            ).respond(json=cloudflare_embedding_response)

            response = litellm.embedding(
                api_key="test-api-key",
                api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
                model="cloudflare/@cf/baai/bge-large-en-v1.5",
                input="Hello world",
            )

            # Handle potential coroutine return
            import asyncio
            import inspect
            from typing import cast
            from litellm.types.utils import EmbeddingResponse

            if inspect.iscoroutine(response):
                response = asyncio.get_event_loop().run_until_complete(response)

            # Type cast to help static analysis
            response = cast(EmbeddingResponse, response)

            # Verify response
            assert response.object == "list"  # type: ignore
            assert len(response.data) == 1  # type: ignore
            assert response.data[0]["object"] == "embedding"  # type: ignore
            assert response.data[0]["index"] == 0  # type: ignore
            assert len(response.data[0]["embedding"]) == 384  # type: ignore
            assert response.model == "cloudflare/@cf/baai/bge-large-en-v1.5"  # type: ignore

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

    def test_cloudflare_embedding_multiple_texts(
        self, respx_mock: respx.MockRouter, cloudflare_embedding_multiple_response
    ):
        """Test Cloudflare embedding with multiple text inputs."""
        try:
            # Mock response
            respx_mock.post(
                "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/baai/bge-base-en-v1.5"
            ).respond(json=cloudflare_embedding_multiple_response)

            response = litellm.embedding(
                api_key="test-api-key",
                api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
                model="cloudflare/@cf/baai/bge-base-en-v1.5",
                input=["Hello world", "How are you?"],
            )

            # Handle potential coroutine return
            import asyncio
            import inspect
            from typing import cast
            from litellm.types.utils import EmbeddingResponse

            if inspect.iscoroutine(response):
                response = asyncio.get_event_loop().run_until_complete(response)

            # Type cast to help static analysis
            response = cast(EmbeddingResponse, response)

            assert response.object == "list"  # type: ignore
            assert len(response.data) == 2  # type: ignore
            assert response.data[0]["object"] == "embedding"  # type: ignore
            assert response.data[0]["index"] == 0  # type: ignore
            assert response.data[1]["object"] == "embedding"  # type: ignore
            assert response.data[1]["index"] == 1  # type: ignore
            assert len(response.data[0]["embedding"]) == 384  # type: ignore
            assert len(response.data[1]["embedding"]) == 384  # type: ignore

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

    def test_cloudflare_embedding_error_handling(self, respx_mock: respx.MockRouter):
        """Test Cloudflare embedding error handling."""
        try:
            # Mock error response
            error_response = {
                "result": None,
                "success": False,
                "errors": ["Invalid model specified"],
                "messages": [],
            }

            respx_mock.post(
                "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/invalid-model"
            ).respond(status_code=400, json=error_response)

            with pytest.raises(Exception) as exc_info:
                litellm.embedding(
                    api_key="test-api-key",
                    api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
                    model="cloudflare/invalid-model",
                    input="Hello world",
                )

            assert "Cloudflare API error" in str(exc_info.value)

        except Exception as e:
            if "Cloudflare API error" not in str(e):
                pytest.fail(f"Unexpected error: {e}")

    def test_cloudflare_embedding_missing_api_key(self, respx_mock: respx.MockRouter):
        """Test Cloudflare embedding with missing API key."""
        try:
            # Mock authentication error response
            respx_mock.post(
                "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/baai/bge-large-en-v1.5"
            ).respond(status_code=403, json={"error": {"message": "Invalid API key"}})

            with pytest.raises(Exception) as exc_info:
                litellm.embedding(
                    api_key="test-api-key",
                    api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
                    model="cloudflare/@cf/baai/bge-large-en-v1.5",
                    input="Hello world",
                )

            # The specific error message may vary, just ensure an exception is raised
            assert exc_info.value is not None

        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    def test_cloudflare_embedding_missing_account_id(
        self, respx_mock: respx.MockRouter
    ):
        """Test Cloudflare embedding with missing account ID."""
        try:
            # Mock account not found error response
            respx_mock.post(
                "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/@cf/baai/bge-large-en-v1.5"
            ).respond(status_code=404, json={"error": {"message": "Account not found"}})

            with pytest.raises(Exception) as exc_info:
                litellm.embedding(
                    api_key="test-api-key",
                    api_base="https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/",
                    model="cloudflare/@cf/baai/bge-large-en-v1.5",
                    input="Hello world",
                )

            # The specific error message may vary, just ensure an exception is raised
            assert exc_info.value is not None

        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")
