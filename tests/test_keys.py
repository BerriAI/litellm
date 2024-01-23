# What this tests ?
## Tests /key endpoints.

import pytest
import asyncio
import aiohttp
from openai import AsyncOpenAI
import sys, os

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm


async def generate_key(session, i):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["azure-models", "gpt-4"],
        "aliases": {"mistral-7b": "gpt-3.5-turbo"},
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


@pytest.mark.asyncio
async def test_key_gen():
    async with aiohttp.ClientSession() as session:
        tasks = [generate_key(session, i) for i in range(1, 11)]
        await asyncio.gather(*tasks)


async def update_key(session, get_key):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/key/update"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {"key": get_key, "models": ["gpt-4"]}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def chat_completion(session, key, model="gpt-4"):
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

        return await response.json()


async def chat_completion_streaming(session, key, model="gpt-4"):
    client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    data = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    response = await client.chat.completions.create(**data)

    content = ""
    async for chunk in response:
        content += chunk.choices[0].delta.content or ""

    print(f"content: {content}")
    prompt_tokens = litellm.token_counter(model="azure/gpt-35-turbo", messages=messages)
    completion_tokens = litellm.token_counter(
        model="azure/gpt-35-turbo", text=content, count_response_tokens=True
    )

    return prompt_tokens, completion_tokens


@pytest.mark.asyncio
async def test_key_update():
    """
    Create key
    Update key with new model
    Test key w/ model
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        await update_key(
            session=session,
            get_key=key,
        )
        await chat_completion(session=session, key=key)


async def delete_key(session, get_key):
    """
    Delete key
    """
    url = "http://0.0.0.0:4000/key/delete"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {"keys": [get_key]}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_key_delete():
    """
    Delete key
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        await delete_key(
            session=session,
            get_key=key,
        )


async def get_key_info(session, get_key, call_key):
    """
    Make sure only models user has access to are returned
    """
    url = f"http://0.0.0.0:4000/key/info?key={get_key}"
    headers = {
        "Authorization": f"Bearer {call_key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            if call_key != get_key:
                return status
            else:
                print(f"call_key: {call_key}; get_key: {get_key}")
                raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_key_info():
    """
    Get key info
    - as admin -> 200
    - as key itself -> 200
    - as random key -> 403
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        # as admin #
        await get_key_info(session=session, get_key=key, call_key="sk-1234")
        # as key itself #
        await get_key_info(session=session, get_key=key, call_key=key)
        # as random key #
        key_gen = await generate_key(session=session, i=0)
        random_key = key_gen["key"]
        status = await get_key_info(session=session, get_key=key, call_key=random_key)
        assert status == 403


@pytest.mark.asyncio
async def test_key_info_spend_values():
    """
    - create key
    - make completion call
    - assert cost is expected value
    """
    async with aiohttp.ClientSession() as session:
        ## Test Spend Update ##
        # completion
        # response = await chat_completion(session=session, key=key)
        # prompt_cost, completion_cost = litellm.cost_per_token(
        #     model="azure/gpt-35-turbo",
        #     prompt_tokens=response["usage"]["prompt_tokens"],
        #     completion_tokens=response["usage"]["completion_tokens"],
        # )
        # response_cost = prompt_cost + completion_cost
        # await asyncio.sleep(5)  # allow db log to be updated
        # key_info = await get_key_info(session=session, get_key=key, call_key=key)
        # print(
        #     f"response_cost: {response_cost}; key_info spend: {key_info['info']['spend']}"
        # )
        # assert response_cost == key_info["info"]["spend"]
        ## streaming
        key_gen = await generate_key(session=session, i=0)
        new_key = key_gen["key"]
        prompt_tokens, completion_tokens = await chat_completion_streaming(
            session=session, key=new_key
        )
        print(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="azure/gpt-35-turbo",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        response_cost = prompt_cost + completion_cost
        await asyncio.sleep(5)  # allow db log to be updated
        key_info = await get_key_info(
            session=session, get_key=new_key, call_key=new_key
        )
        print(
            f"response_cost: {response_cost}; key_info spend: {key_info['info']['spend']}"
        )
        assert response_cost == key_info["info"]["spend"]
