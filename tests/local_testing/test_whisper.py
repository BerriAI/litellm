# What is this?
## Tests `litellm.transcription` endpoint. Outside litellm module b/c of audio file used in testing (it's ~700kb).

import asyncio
import logging
import os
import sys
import time
import traceback
from typing import Optional

import aiohttp
import dotenv
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI

import litellm
from litellm.integrations.custom_logger import CustomLogger

# Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

audio_file = open(file_path, "rb")


file2_path = os.path.join(pwd, "eagle.wav")
audio_file2 = open(file2_path, "rb")

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router


@pytest.mark.parametrize(
    "model, api_key, api_base",
    [
        ("whisper-1", None, None),
        # ("groq/whisper-large-v3", None, None),
        (
            "azure/azure-whisper",
            os.getenv("AZURE_EUROPE_API_KEY"),
            "https://my-endpoint-europe-berri-992.openai.azure.com/",
        ),
    ],
)
@pytest.mark.parametrize("response_format", ["json", "vtt"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_transcription(model, api_key, api_base, response_format, sync_mode):
    if sync_mode:
        transcript = litellm.transcription(
            model=model,
            file=audio_file,
            api_key=api_key,
            api_base=api_base,
            response_format=response_format,
            drop_params=True,
        )
    else:
        transcript = await litellm.atranscription(
            model=model,
            file=audio_file,
            api_key=api_key,
            api_base=api_base,
            response_format=response_format,
            drop_params=True,
        )
    print(f"transcript: {transcript.model_dump()}")
    print(f"transcript: {transcript._hidden_params}")

    assert transcript.text is not None


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

        except:
            pass


proxy_handler_instance = MyCustomHandler()


# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy
@pytest.mark.asyncio
async def test_transcription_on_router():
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

        response = await router.atranscription(
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


@pytest.mark.asyncio()
async def test_transcription_caching():
    import litellm
    from litellm.caching import Cache

    litellm.set_verbose = True
    litellm.cache = Cache()

    # make raw llm api call

    response_1 = await litellm.atranscription(
        model="whisper-1",
        file=audio_file,
    )

    await asyncio.sleep(5)

    # cache hit

    response_2 = await litellm.atranscription(
        model="whisper-1",
        file=audio_file,
    )

    print("response_1", response_1)
    print("response_2", response_2)
    print("response2 hidden params", response_2._hidden_params)
    assert response_2._hidden_params["cache_hit"] is True

    # cache miss

    response_3 = await litellm.atranscription(
        model="whisper-1",
        file=audio_file2,
    )
    print("response_3", response_3)
    print("response3 hidden params", response_3._hidden_params)
    assert response_3._hidden_params.get("cache_hit") is not True
    assert response_3.text != response_2.text

    litellm.cache = None
