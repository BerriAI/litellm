# What this tests ?
## Tests /key endpoints.

import pytest
import asyncio
import aiohttp


async def generate_key(session, i):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["azure-models"],
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
