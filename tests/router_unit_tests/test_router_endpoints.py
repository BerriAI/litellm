import sys
import os
import json
import traceback
from typing import Optional
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router, CustomLogger
from litellm.types.utils import StandardLoggingPayload

# Get the current directory of the file being run
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
async def test_anthropic_router_completion_e2e(model_list):
    from litellm.adapters.anthropic_adapter import anthropic_adapter
    from litellm.types.llms.anthropic import AnthropicResponse

    litellm.set_verbose = True

    litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    ## Test 1: user facing function
    response = await router.aadapter_completion(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
        adapter_id="anthropic",
        mock_response="This is a fake call",
    )

    ## Test 2: underlying function
    await router._aadapter_completion(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
        adapter_id="anthropic",
        mock_response="This is a fake call",
    )

    print("Response: {}".format(response))

    assert response is not None

    AnthropicResponse.model_validate(response)

    assert response.model == "gpt-3.5-turbo"


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
