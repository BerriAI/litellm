import sys
import os
import json
import traceback
from typing import Optional
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router, CustomLogger
from litellm.types.utils import StandardLoggingPayload

## Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

audio_file = open(file_path, "rb")
from pathlib import Path
import litellm
import pytest
import asyncio


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "dall-e-3",
            "litellm_params": {
                "model": "dall-e-3",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "cohere-rerank",
            "litellm_params": {
                "model": "cohere/rerank-english-v3.0",
                "api_key": os.getenv("COHERE_API_KEY"),
            },
        },
        {
            "model_name": "claude-3-5-sonnet-20240620",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "mock_response": "hi this is macintosh.",
            },
        },
    ]


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger):
    def __init__(self):
        self.openai_client = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            # init logging config
            print("logging a transcript kwargs: ", kwargs)
            print("openai client=", kwargs.get("client"))
            self.openai_client = kwargs.get("client")
            self.standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )

        except Exception:
            pass


# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy
@pytest.mark.asyncio
@pytest.mark.flaky(retries=6, delay=10)
async def test_transcription_on_router():
    proxy_handler_instance = MyCustomHandler()
    litellm.set_verbose = True
    litellm.callbacks = [proxy_handler_instance]
    print("\n Testing async transcription on router\n")
    try:
        model_list = [
            {
                "model_name": "whisper",
                "litellm_params": {
                    "model": "whisper-1",
                },
            },
            {
                "model_name": "whisper",
                "litellm_params": {
                    "model": "azure/azure-whisper",
                    "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com/",
                    "api_key": os.getenv("AZURE_EUROPE_API_KEY"),
                    "api_version": "2024-02-15-preview",
                },
            },
        ]

        router = Router(model_list=model_list)

        router_level_clients = []
        for deployment in router.model_list:
            _deployment_openai_client = router._get_client(
                deployment=deployment,
                kwargs={"model": "whisper-1"},
                client_type="async",
            )

            router_level_clients.append(str(_deployment_openai_client))

        ## test 1: user facing function
        response = await router.atranscription(
            model="whisper",
            file=audio_file,
        )

        ## test 2: underlying function
        response = await router._atranscription(
            model="whisper",
            file=audio_file,
        )
        print(response)

        # PROD Test
        # Ensure we ONLY use OpenAI/Azure client initialized on the router level
        await asyncio.sleep(5)
        print("OpenAI Client used= ", proxy_handler_instance.openai_client)
        print("all router level clients= ", router_level_clients)
        assert proxy_handler_instance.openai_client in router_level_clients
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("mode", ["iterator"])  # "file",
@pytest.mark.asyncio
async def test_audio_speech_router(mode):
    litellm.set_verbose = True
    test_logger = MyCustomHandler()
    litellm.callbacks = [test_logger]
    from litellm import Router

    client = Router(
        model_list=[
            {
                "model_name": "tts",
                "litellm_params": {
                    "model": "openai/tts-1",
                },
            },
        ]
    )

    response = await client.aspeech(
        model="tts",
        voice="alloy",
        input="the quick brown fox jumped over the lazy dogs",
        api_base=None,
        api_key=None,
        organization=None,
        project=None,
        max_retries=1,
        timeout=600,
        client=None,
        optional_params={},
    )

    await asyncio.sleep(3)

    from litellm.llms.openai.openai import HttpxBinaryResponseContent

    assert isinstance(response, HttpxBinaryResponseContent)

    assert test_logger.standard_logging_object is not None
    print(
        "standard_logging_object=",
        json.dumps(test_logger.standard_logging_object, indent=4),
    )
    assert test_logger.standard_logging_object["model_group"] == "tts"


@pytest.mark.asyncio()
async def test_rerank_endpoint(model_list):
    from litellm.types.utils import RerankResponse

    router = Router(model_list=model_list)

    ## Test 1: user facing function
    response = await router.arerank(
        model="cohere-rerank",
        query="hello",
        documents=["hello", "world"],
        top_n=3,
    )

    ## Test 2: underlying function
    response = await router._arerank(
        model="cohere-rerank",
        query="hello",
        documents=["hello", "world"],
        top_n=3,
    )

    print("async re rank response: ", response)

    assert response.id is not None
    assert response.results is not None

    RerankResponse.model_validate(response)


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "model", ["omni-moderation-latest", "openai/omni-moderation-latest", None]
)
async def test_moderation_endpoint(model):
    litellm.set_verbose = True
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                },
            },
            {
                "model_name": "*",
                "litellm_params": {
                    "model": "openai/*",
                },
            },
        ]
    )

    if model is None:
        response = await router.amoderation(input="hello this is a test")
    else:
        response = await router.amoderation(model=model, input="hello this is a test")

    print("moderation response: ", response)


@pytest.mark.asyncio()
async def test_moderation_endpoint_with_api_base():
    """
    Test that the moderation endpoint respects api_base configuration
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    custom_api_base = "https://us.api.openai.com/v1"
    
    router = Router(
        model_list=[
            {
                "model_name": "openai/omni-moderation-latest",
                "litellm_params": {
                    "model": "openai/omni-moderation-latest",
                    "api_base": custom_api_base,
                    "api_key": "test-key"
                },
            },
        ]
    )

    # Mock the OpenAI client to verify api_base is passed
    with patch("litellm.main.openai_chat_completions._get_openai_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "modr-123",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": False,
                    "categories": {},
                    "category_scores": {},
                    "category_applied_input_types": {}
                }
            ]
        }
        mock_client.moderations.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        response = await router.amoderation(
            model="openai/omni-moderation-latest",
            input="hello this is a test"
        )
        
        # Verify that _get_openai_client was called with the custom api_base
        mock_get_client.assert_called()
        call_kwargs = mock_get_client.call_args.kwargs
        assert call_kwargs.get("api_base") == custom_api_base, \
            f"Expected api_base to be {custom_api_base}, but got {call_kwargs.get('api_base')}"
        
        print(f"✓ Moderation endpoint correctly uses api_base: {custom_api_base}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_aaaaatext_completion_endpoint(model_list, sync_mode):
    router = Router(model_list=model_list)

    if sync_mode:
        response = router.text_completion(
            model="gpt-3.5-turbo",
            prompt="Hello, how are you?",
            mock_response="I'm fine, thank you!",
        )
    else:
        ## Test 1: user facing function
        response = await router.atext_completion(
            model="gpt-3.5-turbo",
            prompt="Hello, how are you?",
            mock_response="I'm fine, thank you!",
        )

        ## Test 2: underlying function
        response_2 = await router._atext_completion(
            model="gpt-3.5-turbo",
            prompt="Hello, how are you?",
            mock_response="I'm fine, thank you!",
        )
        assert response_2.choices[0].text == "I'm fine, thank you!"

    assert response.choices[0].text == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_with_empty_choices(model_list):
    """
    https://github.com/BerriAI/litellm/issues/8306
    """
    router = Router(model_list=model_list)
    mock_response = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        ),
        model="gpt-3.5-turbo",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response=mock_response,
    )
    assert response is not None


@pytest.mark.parametrize("sync_mode", [True, False])
def test_generic_api_call_with_fallbacks_basic(sync_mode):
    """
    Test both the sync and async versions of generic_api_call_with_fallbacks with a basic successful call
    """
    # Create a mock function that will be passed to generic_api_call_with_fallbacks
    if sync_mode:
        from unittest.mock import Mock

        mock_function = Mock()
        mock_function.__name__ = "test_function"
    else:
        mock_function = AsyncMock()
        mock_function.__name__ = "test_function"

    # Create a mock response
    mock_response = {
        "id": "resp_123456",
        "role": "assistant",
        "content": "This is a test response",
        "model": "test-model",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    mock_function.return_value = mock_response

    # Create a router with a test model
    router = Router(
        model_list=[
            {
                "model_name": "test-model-alias",
                "litellm_params": {
                    "model": "anthropic/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    # Call the appropriate generic_api_call_with_fallbacks method
    if sync_mode:
        response = router._generic_api_call_with_fallbacks(
            model="test-model-alias",
            original_function=mock_function,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
    else:
        response = asyncio.run(
            router._ageneric_api_call_with_fallbacks(
                model="test-model-alias",
                original_function=mock_function,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )
        )

    # Verify the mock function was called
    mock_function.assert_called_once()

    # Verify the response
    assert response == mock_response


@pytest.mark.asyncio
async def test_aadapter_completion():
    """
    Test the aadapter_completion method which uses async_function_with_fallbacks
    """
    # Create a mock for the _aadapter_completion method
    mock_response = {
        "id": "adapter_resp_123",
        "object": "adapter.completion",
        "created": 1677858242,
        "model": "test-model-with-adapter",
        "choices": [
            {
                "text": "This is a test adapter response",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    # Create a router with a patched _aadapter_completion method
    with patch.object(
        Router, "_aadapter_completion", new_callable=AsyncMock
    ) as mock_method:
        mock_method.return_value = mock_response

        router = Router(
            model_list=[
                {
                    "model_name": "test-adapter-model",
                    "litellm_params": {
                        "model": "anthropic/test-model",
                        "api_key": "fake-api-key",
                    },
                }
            ]
        )

        # Replace the async_function_with_fallbacks with a mock
        router.async_function_with_fallbacks = AsyncMock(return_value=mock_response)

        # Call the aadapter_completion method
        response = await router.aadapter_completion(
            adapter_id="test-adapter-id",
            model="test-adapter-model",
            prompt="This is a test prompt",
            max_tokens=100,
        )

        # Verify the response
        assert response == mock_response

        # Verify async_function_with_fallbacks was called with the right parameters
        router.async_function_with_fallbacks.assert_called_once()
        call_kwargs = router.async_function_with_fallbacks.call_args.kwargs
        assert call_kwargs["adapter_id"] == "test-adapter-id"
        assert call_kwargs["model"] == "test-adapter-model"
        assert call_kwargs["prompt"] == "This is a test prompt"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["original_function"] == router._aadapter_completion
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["model_group"] == "test-adapter-model"


@pytest.mark.asyncio
async def test__aadapter_completion():
    """
    Test the _aadapter_completion method directly
    """
    # Create a mock response for litellm.aadapter_completion
    mock_response = {
        "id": "adapter_resp_123",
        "object": "adapter.completion",
        "created": 1677858242,
        "model": "test-model-with-adapter",
        "choices": [
            {
                "text": "This is a test adapter response",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    # Create a router with a mocked litellm.aadapter_completion
    with patch(
        "litellm.aadapter_completion", new_callable=AsyncMock
    ) as mock_adapter_completion:
        mock_adapter_completion.return_value = mock_response

        router = Router(
            model_list=[
                {
                    "model_name": "test-adapter-model",
                    "litellm_params": {
                        "model": "anthropic/test-model",
                        "api_key": "fake-api-key",
                    },
                }
            ]
        )

        # Mock the async_get_available_deployment method
        router.async_get_available_deployment = AsyncMock(
            return_value={
                "model_name": "test-adapter-model",
                "litellm_params": {
                    "model": "test-model",
                    "api_key": "fake-api-key",
                },
                "model_info": {
                    "id": "test-unique-id",
                },
            }
        )

        # Mock the async_routing_strategy_pre_call_checks method
        router.async_routing_strategy_pre_call_checks = AsyncMock()

        # Call the _aadapter_completion method
        response = await router._aadapter_completion(
            adapter_id="test-adapter-id",
            model="test-adapter-model",
            prompt="This is a test prompt",
            max_tokens=100,
        )

        # Verify the response
        assert response == mock_response

        # Verify litellm.aadapter_completion was called with the right parameters
        mock_adapter_completion.assert_called_once()
        call_kwargs = mock_adapter_completion.call_args.kwargs
        assert call_kwargs["adapter_id"] == "test-adapter-id"
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["prompt"] == "This is a test prompt"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["api_key"] == "fake-api-key"
        assert call_kwargs["caching"] == router.cache_responses

        # Verify the success call was recorded
        assert router.success_calls["test-model"] == 1
        assert router.total_calls["test-model"] == 1

        # Verify async_routing_strategy_pre_call_checks was called
        router.async_routing_strategy_pre_call_checks.assert_called_once()


def test_initialize_router_endpoints():
    """
    Test that initialize_router_endpoints correctly sets up all router endpoints
    """
    # Create a router with a basic model
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "anthropic/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    # Explicitly call initialize_router_endpoints
    router.initialize_router_endpoints()

    # Verify all expected endpoints are initialized
    assert hasattr(router, "amoderation")
    assert hasattr(router, "aanthropic_messages")
    assert hasattr(router, "aresponses")
    assert hasattr(router, "responses")
    assert hasattr(router, "aget_responses")
    assert hasattr(router, "adelete_responses")
    # Verify the endpoints are callable
    assert callable(router.amoderation)
    assert callable(router.aanthropic_messages)
    assert callable(router.aresponses)
    assert callable(router.responses)
    assert callable(router.aget_responses)
    assert callable(router.adelete_responses)


@pytest.mark.asyncio
async def test_init_responses_api_endpoints():
    """
    A simpler test for _init_responses_api_endpoints that focuses on the basic functionality
    """
    from litellm.responses.utils import ResponsesAPIRequestUtils
    # Create a router with a basic model
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )
    
    # Just mock the _ageneric_api_call_with_fallbacks method
    router._ageneric_api_call_with_fallbacks = AsyncMock()
    
    # Add a mock implementation of _get_model_id_from_response_id to the Router instance
    ResponsesAPIRequestUtils.get_model_id_from_response_id = MagicMock(return_value=None)
    
    # Call without a response_id (no model extraction should happen)
    await router._init_responses_api_endpoints(
        original_function=AsyncMock(),
        thread_id="thread_xyz"
    )
    
    # Verify _ageneric_api_call_with_fallbacks was called but model wasn't changed
    first_call_kwargs = router._ageneric_api_call_with_fallbacks.call_args.kwargs
    assert "model" not in first_call_kwargs
    assert first_call_kwargs["thread_id"] == "thread_xyz"
    
    # Reset the mock
    router._ageneric_api_call_with_fallbacks.reset_mock()
    
    # Change the return value for the second call
    ResponsesAPIRequestUtils.get_model_id_from_response_id.return_value = "claude-3-sonnet"
    
    # Call with a response_id
    await router._init_responses_api_endpoints(
        original_function=AsyncMock(),
        response_id="resp_claude_123"
    )
    
    # Verify model was updated in the kwargs
    second_call_kwargs = router._ageneric_api_call_with_fallbacks.call_args.kwargs
    assert second_call_kwargs["model"] == "claude-3-sonnet"
    assert second_call_kwargs["response_id"] == "resp_claude_123"


@pytest.mark.asyncio
async def test_init_vector_store_api_endpoints():
    """
    Test that _init_vector_store_api_endpoints correctly passes custom_llm_provider to kwargs
    """
    # Create a router with a basic model
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )
    
    # Mock the original function
    mock_original_function = AsyncMock(return_value={"status": "success"})
    
    # Call without custom_llm_provider
    result = await router._init_vector_store_api_endpoints(
        original_function=mock_original_function,
        vector_store_id="test-store"
    )
    
    # Verify original function was called with correct kwargs
    mock_original_function.assert_called_once_with(vector_store_id="test-store")
    assert result == {"status": "success"}
    
    # Reset the mock
    mock_original_function.reset_mock()
    
    # Call with custom_llm_provider
    await router._init_vector_store_api_endpoints(
        original_function=mock_original_function,
        custom_llm_provider="openai",
        vector_store_id="test-store"
    )
    
    # Verify custom_llm_provider was added to kwargs
    mock_original_function.assert_called_once_with(
        vector_store_id="test-store",
        custom_llm_provider="openai"
    )


def test_apply_default_settings():
    """
    Test the apply_default_settings method.
    
    This test verifies that apply_default_settings correctly initializes
    default pre-call checks and doesn't modify existing router state.
    """
    # Test with fresh router
    router = Router()
    initial_optional_callbacks = router.optional_callbacks
    
    # Test that the method runs without error
    result = router.apply_default_settings()
    
    # Verify method returns None as expected
    assert result is None
    
    # Verify that optional_callbacks remains None if it was initially None
    # (since default_pre_call_checks is an empty list)
    assert router.optional_callbacks == initial_optional_callbacks
    
    # Test with router that already has some optional_callbacks
    router_with_callbacks = Router()
    mock_callback = MagicMock()
    router_with_callbacks.optional_callbacks = [mock_callback]
    
    # Apply default settings
    result = router_with_callbacks.apply_default_settings()
    
    # Verify method returns None
    assert result is None
    
    # Verify existing callbacks are preserved (since we're adding empty list)
    assert mock_callback in router_with_callbacks.optional_callbacks
    
    # Test that the method is called during router initialization
    with patch.object(Router, 'apply_default_settings') as mock_apply:
        Router()
        mock_apply.assert_called_once()
    
    # Test with mocked add_optional_pre_call_checks to verify internal call
    router_test = Router()
    with patch.object(router_test, 'add_optional_pre_call_checks') as mock_add_checks:
        router_test.apply_default_settings()
        
        # Verify add_optional_pre_call_checks was called with empty list
        mock_add_checks.assert_called_once_with([])




def test_initialize_core_endpoints():
    """
    Test that _initialize_core_endpoints correctly sets up all core router endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "anthropic/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_core_endpoints()

    core_endpoints = [
        "amoderation",
        "aanthropic_messages",
        "agenerate_content",
        "aadapter_generate_content",
        "aresponses",
        "afile_delete",
        "afile_content",
        "responses",
        "aget_responses",
        "acancel_responses",
        "adelete_responses",
        "alist_input_items",
        "_arealtime",
        "acreate_fine_tuning_job",
        "acancel_fine_tuning_job",
        "alist_fine_tuning_jobs",
        "aretrieve_fine_tuning_job",
        "afile_list",
        "aimage_edit",
        "allm_passthrough_route",
    ]

    for endpoint in core_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_specialized_endpoints():
    """
    Test that _initialize_specialized_endpoints correctly sets up specialized endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_specialized_endpoints()

    specialized_endpoints = [
        "avector_store_search",
        "avector_store_create",
        "vector_store_search",
        "vector_store_create",
        "agenerate_content",
        "generate_content",
        "agenerate_content_stream",
        "generate_content_stream",
        "aocr",
        "ocr",
        "asearch",
        "search",
        "avideo_generation",
        "video_generation",
        "avideo_list",
        "video_list",
        "avideo_status",
        "video_status",
        "avideo_content",
        "video_content",
        "avideo_remix",
        "video_remix",
        "acreate_container",
        "create_container",
        "alist_containers",
        "list_containers",
        "aretrieve_container",
        "retrieve_container",
        "adelete_container",
        "delete_container",
        "acreate_skill",
        "alist_skills",
        "aget_skill",
        "adelete_skill",
    ]

    for endpoint in specialized_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_vector_store_endpoints():
    """
    Test that _initialize_vector_store_endpoints correctly sets up vector store endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_vector_store_endpoints()

    vector_store_endpoints = [
        "avector_store_search",
        "avector_store_create",
        "vector_store_search",
        "vector_store_create",
    ]

    for endpoint in vector_store_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_vector_store_file_endpoints():
    """
    Test that _initialize_vector_store_file_endpoints correctly sets up vector store file endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_vector_store_file_endpoints()

    vector_store_file_endpoints = [
        "avector_store_file_create",
        "vector_store_file_create",
        "avector_store_file_list",
        "vector_store_file_list",
        "avector_store_file_retrieve",
        "vector_store_file_retrieve",
        "avector_store_file_content",
        "vector_store_file_content",
        "avector_store_file_update",
        "vector_store_file_update",
        "avector_store_file_delete",
        "vector_store_file_delete",
    ]

    for endpoint in vector_store_file_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_google_genai_endpoints():
    """
    Test that _initialize_google_genai_endpoints correctly sets up Google GenAI endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_google_genai_endpoints()

    google_genai_endpoints = [
        "agenerate_content",
        "generate_content",
        "agenerate_content_stream",
        "generate_content_stream",
    ]

    for endpoint in google_genai_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_ocr_search_endpoints():
    """
    Test that _initialize_ocr_search_endpoints correctly sets up OCR and search endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_ocr_search_endpoints()

    ocr_search_endpoints = [
        "aocr",
        "ocr",
        "asearch",
        "search",
    ]

    for endpoint in ocr_search_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_video_endpoints():
    """
    Test that _initialize_video_endpoints correctly sets up video endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_video_endpoints()

    video_endpoints = [
        "avideo_generation",
        "video_generation",
        "avideo_list",
        "video_list",
        "avideo_status",
        "video_status",
        "avideo_content",
        "video_content",
        "avideo_remix",
        "video_remix",
    ]

    for endpoint in video_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_container_endpoints():
    """
    Test that _initialize_container_endpoints correctly sets up container endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_container_endpoints()

    container_endpoints = [
        "acreate_container",
        "create_container",
        "alist_containers",
        "list_containers",
        "aretrieve_container",
        "retrieve_container",
        "adelete_container",
        "delete_container",
    ]

    for endpoint in container_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_skills_endpoints():
    """
    Test that _initialize_skills_endpoints correctly sets up skills endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "anthropic/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router._initialize_skills_endpoints()

    skills_endpoints = [
        "acreate_skill",
        "alist_skills",
        "aget_skill",
        "adelete_skill",
    ]

    for endpoint in skills_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


@pytest.mark.asyncio
async def test_init_containers_api_endpoints():
    """
    Test that _init_containers_api_endpoints calls the original function
    directly without model-based routing.
    """
    router = Router(model_list=[])

    mock_response = {"id": "cntr_test", "name": "Test Container"}
    mock_original_function = AsyncMock(return_value=mock_response)

    result = await router._init_containers_api_endpoints(
        original_function=mock_original_function,
        custom_llm_provider="openai",
        name="Test Container"
    )

    mock_original_function.assert_called_once_with(
        custom_llm_provider="openai",
        name="Test Container"
    )
    assert result == mock_response
