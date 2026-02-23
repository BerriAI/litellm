# What this tests ?
## Tests /chat/completions by generating a key and then making a chat completions-request
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from typing import Optional, List, Union
from litellm._uuid import uuid

LITELLM_MASTER_KEY = "sk-1234"


def response_header_check(response):
    """
    - assert if response headers < 4kb (nginx limit).
    """
    headers_size = sum(len(k) + len(v) for k, v in response.raw_headers)
    assert headers_size < 4096, "Response headers exceed the 4kb limit"


async def generate_key(
    session,
    models=[
        "gpt-4",
        "text-embedding-ada-002",
        "dall-e-2",
        "fake-openai-endpoint-2",
        "mistral-embed",
    ],
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers

        return await response.json()


async def new_user(session):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["gpt-4", "text-embedding-ada-002", "dall-e-2"],
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers
        return await response.json()


async def moderation(session, key):
    url = "http://0.0.0.0:4000/moderations"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {"input": "I want to kill the cat."}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def chat_completion(session, key, model: Union[str, List] = "gpt-4"):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Hello! {uuid.uuid4()}"},
        ],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(
                f"Request did not return a 200 status code: {status}, response text={response_text}"
            )

        response_header_check(
            response
        )  # calling the function to check response headers

        return await response.json()


async def queue_chat_completion(
    session, key, priority: int, model: Union[str, List] = "gpt-4"
):
    url = "http://0.0.0.0:4000/queue/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "priority": priority,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return response.raw_headers


async def chat_completion_with_headers(session, key, model="gpt-4"):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers

        raw_headers = response.raw_headers
        raw_headers_json = {}

        for (
            item
        ) in (
            response.raw_headers
        ):  # ((b'date', b'Fri, 19 Apr 2024 21:17:29 GMT'), (), )
            raw_headers_json[item[0].decode("utf-8")] = item[1].decode("utf-8")

        return raw_headers_json


async def chat_completion_with_model_from_route(session, key, route):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def completion(session, key):
    url = "http://0.0.0.0:4000/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {"model": "gpt-4", "prompt": "Hello!"}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers

        response = await response.json()

        return response


async def embeddings(session, key, model="text-embedding-ada-002"):
    url = "http://0.0.0.0:4000/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "input": ["hello world"],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers


async def image_generation(session, key):
    url = "http://0.0.0.0:4000/images/generations"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "dall-e-2",
        "prompt": "A cute baby sea otter",
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            if (
                "Connection error" in response_text
            ):  # OpenAI endpoint returns a connection error
                return
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers


@pytest.mark.asyncio
async def test_chat_completion():
    """
    - Create key
    Make chat completion call
    - Create user
    make chat completion call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, models=["gpt-3.5-turbo"])
        azure_client = AsyncAzureOpenAI(
            azure_endpoint="http://0.0.0.0:4000",
            azure_deployment="random-model",
            api_key=key_gen["key"],
            api_version="2024-02-15-preview",
        )
        with pytest.raises(openai.AuthenticationError) as e:
            response = await azure_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
            )
        assert "key not allowed to access model." in str(e)


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.skip(reason="Flaky test, this works locally but not on CI")
async def test_chat_completion_ratelimit():
    """
    - call model with rpm 1
    - make 2 parallel calls
    - make sure 1 fails
    """
    async with aiohttp.ClientSession() as session:
        # key_gen = await generate_key(session=session)
        key = "sk-1234"
        tasks = []
        tasks.append(
            chat_completion(session=session, key=key, model="fake-openai-endpoint-2")
        )
        tasks.append(
            chat_completion(session=session, key=key, model="fake-openai-endpoint-2")
        )
        try:
            await asyncio.gather(*tasks)
            pytest.fail("Expected at least 1 call to fail")
        except Exception as e:
            if "Request did not return a 200 status code: 429" in str(e):
                pass
            else:
                pytest.fail(f"Wrong error received - {str(e)}")


@pytest.mark.asyncio
@pytest.mark.skip(reason="Flaky test")
async def test_chat_completion_different_deployments():
    """
    - call model group with 2 deployments
    - make 5 calls
    - expect 2 unique deployments
    """
    async with aiohttp.ClientSession() as session:
        # key_gen = await generate_key(session=session)
        key = "sk-1234"
        results = []
        for _ in range(20):
            results.append(
                await chat_completion_with_headers(
                    session=session, key=key, model="fake-openai-endpoint-3"
                )
            )
        try:
            print(f"results: {results}")
            init_model_id = results[0]["x-litellm-model-id"]
            deployments_shuffled = False
            for result in results[1:]:
                if init_model_id != result["x-litellm-model-id"]:
                    deployments_shuffled = True
            if deployments_shuffled == False:
                pytest.fail("Expected at least 1 shuffled call")
        except Exception as e:
            pass


@pytest.mark.asyncio
async def test_chat_completion_streaming():
    """
    [PROD Test] Ensures logprobs are returned correctly
    """
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-large",
        messages=[{"role": "user", "content": "Hello!"}],
        logprobs=True,
        top_logprobs=2,
        stream=True,
    )

    response_str = ""

    async for chunk in response:
        response_str += chunk.choices[0].delta.content or ""

    print(f"response_str: {response_str}")


@pytest.mark.asyncio
async def test_completion_streaming_usage_metrics():
    """
    [PROD Test] Ensures usage metrics are returned correctly when `include_usage` is set to `True`
    """
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

    response = await client.completions.create(
        model="gpt-instruct",
        prompt="hey",
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=4,
        temperature=0.00000001,
    )

    last_chunk = None
    async for chunk in response:
        print("chunk", chunk)
        last_chunk = chunk

    assert last_chunk is not None, "No chunks were received"
    assert last_chunk.usage is not None, "Usage information was not received"
    assert last_chunk.usage.prompt_tokens > 0, "Prompt tokens should be greater than 0"
    assert (
        last_chunk.usage.completion_tokens > 0
    ), "Completion tokens should be greater than 0"
    assert last_chunk.usage.total_tokens > 0, "Total tokens should be greater than 0"


@pytest.mark.asyncio
async def test_chat_completion_anthropic_structured_output():
    """
    Ensure nested pydantic output is returned correctly
    """
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    class EventsList(BaseModel):
        events: list[CalendarEvent]

    messages = [
        {"role": "user", "content": "List 5 important events in the XIX century"}
    ]

    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

    res = await client.beta.chat.completions.parse(
        model="bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        response_format=EventsList,
        timeout=60,
    )
    message = res.choices[0].message

    if message.parsed:
        print(message.parsed.events)


@pytest.mark.asyncio
async def test_completion():
    """
    - Create key
    Make chat completion call
    - Create user
    make chat completion call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await completion(session=session, key=key)
        key_gen = await new_user(session=session)
        key_2 = key_gen["key"]
        # response = await completion(session=session, key=key_2)

    ## validate openai format ##
    client = OpenAI(api_key=key_2, base_url="http://0.0.0.0:4000")

    client.completions.create(
        model="gpt-4",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
    )


@pytest.mark.asyncio
async def test_embeddings():
    """
    - Create key
    Make embeddings call
    - Create user
    make embeddings call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await embeddings(session=session, key=key)
        key_gen = await new_user(session=session)
        key_2 = key_gen["key"]
        await embeddings(session=session, key=key_2)

        # embedding request with non OpenAI model
        await embeddings(session=session, key=key, model="mistral-embed")


@pytest.mark.flaky(retries=5, delay=1)
@pytest.mark.asyncio
async def test_image_generation():
    """
    - Create key
    Make embeddings call
    - Create user
    make embeddings call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await image_generation(session=session, key=key)
        key_gen = await new_user(session=session)
        key_2 = key_gen["key"]
        await image_generation(session=session, key=key_2)


@pytest.mark.asyncio
async def test_openai_wildcard_chat_completion():
    """
    - Create key for model = "*" -> this has access to all models
    - proxy_server_config.yaml has model = *
    - Make chat completion call

    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, models=["*"])
        key = key_gen["key"]

        # call chat/completions with a model that the key was not created for + the model is not on the config.yaml
        await chat_completion(session=session, key=key, model="gpt-3.5-turbo-0125")


@pytest.mark.asyncio
async def test_proxy_all_models():
    """
    - proxy_server_config.yaml has model = * / *
    - Make chat completion call
    - groq is NOT defined on /models


    """
    async with aiohttp.ClientSession() as session:
        # call chat/completions with a model that the key was not created for + the model is not on the config.yaml
        await chat_completion(
            session=session, key=LITELLM_MASTER_KEY, model="groq/llama-3.1-8b-instant"
        )

        await chat_completion(
            session=session,
            key=LITELLM_MASTER_KEY,
            model="anthropic/claude-sonnet-4-5-20250929",
        )


@pytest.mark.asyncio
async def test_batch_chat_completions():
    """
    - Make chat completion call using

    """
    async with aiohttp.ClientSession() as session:

        # call chat/completions with a model that the key was not created for + the model is not on the config.yaml
        response = await chat_completion(
            session=session,
            key="sk-1234",
            model="gpt-3.5-turbo,fake-openai-endpoint",
        )

        print(f"response: {response}")

        assert len(response) == 2
        assert isinstance(response, list)


@pytest.mark.asyncio
async def test_moderations_endpoint():
    """
    - Make chat completion call using

    """
    async with aiohttp.ClientSession() as session:

        # call chat/completions with a model that the key was not created for + the model is not on the config.yaml
        response = await moderation(
            session=session,
            key="sk-1234",
        )

        print(f"response: {response}")

        assert "results" in response
