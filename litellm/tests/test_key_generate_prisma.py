# Test the following scenarios:
# 1. Generate a Key, and use it to make a call
# 2. Make a call with invalid key, expect it to fail
# 3. Make a call to a key with invalid model - expect to fail
# 4. Make a call to a key with valid model - expect to pass
# 5. Make a call with key over budget, expect to fail
# 6. Make a streaming chat/completions call with key over budget, expect to fail
# 7. Make a call with an key that never expires, expect to pass
# 8. Make a call with an expired key, expect to fail


# function to call to generate key - async def new_user(data: NewUserRequest):
# function to validate a request - async def user_auth(request: Request):

import sys, os
import traceback
from dotenv import load_dotenv
from fastapi import Request

load_dotenv()
import os, io

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm, asyncio
from litellm.proxy.proxy_server import new_user, user_api_key_auth, user_update
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm._logging import verbose_proxy_logger

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from litellm.proxy._types import NewUserRequest, DynamoDBArgs
from litellm.proxy.utils import DBClient
from starlette.datastructures import URL
from litellm.caching import DualCache

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


request_data = {
    "model": "azure-gpt-3.5",
    "messages": [
        {"role": "user", "content": "this is my new test. respond in 50 lines"}
    ],
}


@pytest.fixture
def prisma_client():
    # Assuming DBClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    # Reset litellm.proxy.proxy_server.prisma_client to None
    litellm.proxy.proxy_server.custom_db_client = None

    return prisma_client


def test_generate_and_call_with_valid_key(prisma_client):
    # 1. Generate a Key, and use it to make a call

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest()
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)

        asyncio.run(test())
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


def test_call_with_invalid_key(prisma_client):
    # 2. Make a call with invalid key, expect it to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            generated_key = "bad-key"
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"}, receive=None)
            request._url = URL(url="/chat/completions")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("got result", result)
            pytest.fail(f"This should have failed!. IT's an invalid key")

        asyncio.run(test())
    except Exception as e:
        print("Got Exception", e)
        print(e.detail)
        assert "Authentication Error" in e.detail
        pass


def test_call_with_invalid_model(prisma_client):
    # 3. Make a call to a key with an invalid model - expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(models=["mistral"])
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            async def return_body():
                return b'{"model": "gemini-pro-vision"}'

            request.body = return_body

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            pytest.fail(f"This should have failed!. IT's an invalid model")

        asyncio.run(test())
    except Exception as e:
        assert (
            e.detail
            == "Authentication Error, API Key not allowed to access model. This token can only access models=['mistral']. Tried to access gemini-pro-vision"
        )
        pass


def test_call_with_valid_model(prisma_client):
    # 4. Make a call to a key with a valid model - expect to pass
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(models=["mistral"])
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            async def return_body():
                return b'{"model": "mistral"}'

            request.body = return_body

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)

        asyncio.run(test())
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


def test_call_with_key_over_budget(prisma_client):
    # 5. Make a call with a key over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(max_budget=0.00001)
            key = await new_user(request)
            print(key)

            generated_key = key.key
            user_id = key.user_id
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)

            # update spend using track_cost callback, make 2nd request, it should fail
            from litellm.proxy.proxy_server import track_cost_callback
            from litellm import ModelResponse, Choices, Message, Usage

            resp = ModelResponse(
                id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
                choices=[
                    Choices(
                        finish_reason=None,
                        index=0,
                        message=Message(
                            content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
                            role="assistant",
                        ),
                    )
                ],
                model="gpt-35-turbo",  # azure always has model written like this
                usage=Usage(prompt_tokens=210, completion_tokens=200, total_tokens=410),
            )
            await track_cost_callback(
                kwargs={
                    "stream": False,
                    "litellm_params": {
                        "metadata": {
                            "user_api_key": generated_key,
                            "user_api_key_user_id": user_id,
                        }
                    },
                },
                completion_response=resp,
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.detail
        assert "Authentication Error, ExceededBudget:" in error_detail
        print(vars(e))


def test_call_with_key_over_budget_stream(prisma_client):
    # 6. Make a call with a key over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    from litellm._logging import verbose_proxy_logger
    import logging

    litellm.set_verbose = True
    verbose_proxy_logger.setLevel(logging.DEBUG)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(max_budget=0.00001)
            key = await new_user(request)
            print(key)

            generated_key = key.key
            user_id = key.user_id
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)

            # update spend using track_cost callback, make 2nd request, it should fail
            from litellm.proxy.proxy_server import track_cost_callback
            from litellm import ModelResponse, Choices, Message, Usage

            resp = ModelResponse(
                id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
                choices=[
                    Choices(
                        finish_reason=None,
                        index=0,
                        message=Message(
                            content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
                            role="assistant",
                        ),
                    )
                ],
                model="gpt-35-turbo",  # azure always has model written like this
                usage=Usage(prompt_tokens=210, completion_tokens=200, total_tokens=410),
            )
            await track_cost_callback(
                kwargs={
                    "stream": True,
                    "complete_streaming_response": resp,
                    "litellm_params": {
                        "metadata": {
                            "user_api_key": generated_key,
                            "user_api_key_user_id": user_id,
                        }
                    },
                },
                completion_response=ModelResponse(),
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.detail
        assert "Authentication Error, ExceededBudget:" in error_detail
        print(vars(e))


def test_generate_and_call_with_valid_key_never_expires(prisma_client):
    # 7. Make a call with an key that never expires, expect to pass

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(duration=None)
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)

        asyncio.run(test())
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


def test_generate_and_call_with_expired_key(prisma_client):
    # 8. Make a call with an expired key, expect to fail

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(duration="0s")
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. IT's an expired key")

        asyncio.run(test())
    except Exception as e:
        print("Got Exception", e)
        print(e.detail)
        assert "Authentication Error" in e.detail
        pass
