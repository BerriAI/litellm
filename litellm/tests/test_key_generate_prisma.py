# Test the following scenarios:
# 1. Generate a Key, and use it to make a call
# 2. Make a call with invalid key, expect it to fail
# 3. Make a call to a key with invalid model - expect to fail
# 4. Make a call to a key with valid model - expect to pass
# 5. Make a call with user over budget, expect to fail
# 6. Make a streaming chat/completions call with user over budget, expect to fail
# 7. Make a call with an key that never expires, expect to pass
# 8. Make a call with an expired key, expect to fail
# 9. Delete a Key
# 10. Generate a key, call key/info. Assert info returned is the same as generated key info
# 11. Generate a Key, cal key/info, call key/update, call key/info
# 12. Make a call with key over budget, expect to fail
# 14. Make a streaming chat/completions call with key over budget, expect to fail
# 15. Generate key, when `allow_user_auth`=False - check if `/key/info` returns key_name=null
# 16. Generate key, when `allow_user_auth`=True - check if `/key/info` returns key_name=sk...<last-4-digits>


# function to call to generate key - async def new_user(data: NewUserRequest):
# function to validate a request - async def user_auth(request: Request):

import sys, os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

load_dotenv()
import os, io, time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm, asyncio
from litellm.proxy.proxy_server import (
    new_user,
    generate_key_fn,
    user_api_key_auth,
    user_update,
    delete_key_fn,
    info_key_fn,
    update_key_fn,
    generate_key_fn,
    generate_key_helper_fn,
    spend_user_fn,
    spend_key_fn,
    view_spend_logs,
    user_info,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token
from litellm._logging import verbose_proxy_logger

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from litellm.proxy._types import (
    NewUserRequest,
    GenerateKeyRequest,
    DynamoDBArgs,
    KeyRequest,
    UpdateKeyRequest,
    GenerateKeyRequest,
)
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
    from litellm.proxy.proxy_cli import append_query_params

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    # Assuming DBClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    # Reset litellm.proxy.proxy_server.prisma_client to None
    litellm.proxy.proxy_server.custom_db_client = None
    litellm.proxy.proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )
    litellm.proxy.proxy_server.user_custom_key_generate = None

    return prisma_client


def test_generate_and_call_with_valid_key(prisma_client):
    # 1. Generate a Key, and use it to make a call

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            from litellm.proxy.proxy_server import user_api_key_cache

            request = NewUserRequest(user_role="app_owner")
            key = await new_user(request)
            print(key)
            user_id = key.user_id

            # check /user/info to verify user_role was set correctly
            new_user_info = await user_info(user_id=user_id)
            new_user_info = new_user_info["user_info"]
            print("new_user_info=", new_user_info)
            assert new_user_info.user_role == "app_owner"
            assert new_user_info.user_id == user_id

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            assert generated_key not in user_api_key_cache.in_memory_cache.cache_dict

            value_from_prisma = await prisma_client.get_data(
                token=generated_key,
            )
            print("token from prisma", value_from_prisma)

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
        print(e.message)
        assert "Authentication Error" in e.message
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
            e.message
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


def test_call_with_user_over_budget(prisma_client):
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
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )
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
                    "response_cost": 0.00002,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            await asyncio.sleep(5)
            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
        assert "Authentication Error, ExceededBudget:" in error_detail
        print(vars(e))


def test_call_with_end_user_over_budget(prisma_client):
    # Test if a user passed to /chat/completions is tracked & fails whe they cross their budget
    # we only check this when litellm.max_user_budget is set
    import random

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm, "max_user_budget", 0.00001)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = GenerateKeyRequest()  # create a key with no budget
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key
            user = f"ishaan {random.randint(0, 10000)}"
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            async def return_body():
                return_string = f'{{"model": "gemini-pro-vision", "user": "{user}"}}'
                # return string as bytes
                return return_string.encode()

            request.body = return_body

            # update spend using track_cost callback, make 2nd request, it should fail
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )
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
                            "user_api_key_user_id": user,
                        },
                        "proxy_server_request": {
                            "user": user,
                        },
                    },
                    "response_cost": 10,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            await asyncio.sleep(5)
            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
        assert "Authentication Error, ExceededBudget:" in error_detail
        print(vars(e))


def test_call_with_proxy_over_budget(prisma_client):
    # 5.1 Make a call with a proxy over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    litellm_proxy_budget_name = f"litellm-proxy-budget-{time.time()}"
    setattr(
        litellm.proxy.proxy_server,
        "litellm_proxy_budget_name",
        litellm_proxy_budget_name,
    )
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            ## CREATE PROXY + USER BUDGET ##
            request = NewUserRequest(
                max_budget=0.00001, user_id=litellm_proxy_budget_name
            )
            await new_user(request)
            request = NewUserRequest()
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
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )
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
                    "response_cost": 0.00002,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            await asyncio.sleep(5)
            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        if hasattr(e, "message"):
            error_detail = e.message
        else:
            error_detail = traceback.format_exc()
        assert "Authentication Error, ExceededBudget:" in error_detail
        print(vars(e))


def test_call_with_user_over_budget_stream(prisma_client):
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
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )
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
                    "response_cost": 0.00002,
                },
                completion_response=ModelResponse(),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            await asyncio.sleep(5)
            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
        assert "Authentication Error, ExceededBudget:" in error_detail
        print(vars(e))


def test_call_with_proxy_over_budget_stream(prisma_client):
    # 6.1 Make a call with a global proxy over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    litellm_proxy_budget_name = f"litellm-proxy-budget-{time.time()}"
    setattr(
        litellm.proxy.proxy_server,
        "litellm_proxy_budget_name",
        litellm_proxy_budget_name,
    )
    from litellm._logging import verbose_proxy_logger
    import logging

    litellm.set_verbose = True
    verbose_proxy_logger.setLevel(logging.DEBUG)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            ## CREATE PROXY + USER BUDGET ##
            request = NewUserRequest(
                max_budget=0.00001, user_id=litellm_proxy_budget_name
            )
            await new_user(request)
            request = NewUserRequest()
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
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )
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
                    "response_cost": 0.00002,
                },
                completion_response=ModelResponse(),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            await asyncio.sleep(5)
            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
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
        print(e.message)
        assert "Authentication Error" in e.message
        pass


def test_delete_key(prisma_client):
    # 9. Generate a Key, delete it. Check if deletion works fine

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "user_custom_auth", None)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            from litellm.proxy.proxy_server import user_api_key_cache

            request = NewUserRequest()
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            delete_key_request = KeyRequest(keys=[generated_key])

            bearer_token = "Bearer sk-1234"

            request = Request(scope={"type": "http"})
            request._url = URL(url="/key/delete")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print(f"result: {result}")
            result.user_role = "proxy_admin"
            # delete the key
            result_delete_key = await delete_key_fn(
                data=delete_key_request, user_api_key_dict=result
            )
            print("result from delete key", result_delete_key)
            assert result_delete_key == {"deleted_keys": [generated_key]}

            assert generated_key not in user_api_key_cache.in_memory_cache.cache_dict
            assert (
                hash_token(generated_key)
                not in user_api_key_cache.in_memory_cache.cache_dict
            )

        asyncio.run(test())
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


def test_delete_key_auth(prisma_client):
    # 10. Generate a Key, delete it, use it to make a call -> expect fail

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            from litellm.proxy.proxy_server import user_api_key_cache

            request = NewUserRequest()
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            delete_key_request = KeyRequest(keys=[generated_key])

            # delete the key
            bearer_token = "Bearer sk-1234"

            request = Request(scope={"type": "http"})
            request._url = URL(url="/key/delete")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print(f"result: {result}")
            result.user_role = "proxy_admin"

            result_delete_key = await delete_key_fn(
                data=delete_key_request, user_api_key_dict=result
            )

            print("result from delete key", result_delete_key)
            assert result_delete_key == {"deleted_keys": [generated_key]}

            request = Request(scope={"type": "http"}, receive=None)
            request._url = URL(url="/chat/completions")

            assert generated_key not in user_api_key_cache.in_memory_cache.cache_dict
            assert (
                hash_token(generated_key)
                not in user_api_key_cache.in_memory_cache.cache_dict
            )

            # use generated key to auth in
            bearer_token = "Bearer " + generated_key
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("got result", result)
            pytest.fail(f"This should have failed!. IT's an invalid key")

        asyncio.run(test())
    except Exception as e:
        print("Got Exception", e)
        print(e.message)
        assert "Authentication Error" in e.message
        pass


def test_generate_and_call_key_info(prisma_client):
    # 10. Generate a Key, cal key/info

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(
                metadata={"team": "litellm-team3", "project": "litellm-project3"}
            )
            key = await new_user(request)
            print(key)

            generated_key = key.key

            # use generated key to auth in
            result = await info_key_fn(key=generated_key)
            print("result from info_key_fn", result)
            assert result["key"] == generated_key
            print("\n info for key=", result["info"])
            assert result["info"]["max_parallel_requests"] == None
            assert result["info"]["metadata"] == {
                "team": "litellm-team3",
                "project": "litellm-project3",
            }

            # cleanup - delete key
            delete_key_request = KeyRequest(keys=[generated_key])
            bearer_token = "Bearer sk-1234"

            request = Request(scope={"type": "http"})
            request._url = URL(url="/key/delete")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print(f"result: {result}")
            result.user_role = "proxy_admin"

            result_delete_key = await delete_key_fn(
                data=delete_key_request, user_api_key_dict=result
            )

        asyncio.run(test())
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


def test_generate_and_update_key(prisma_client):
    # 11. Generate a Key, cal key/info, call key/update, call key/info
    # Check if data gets updated
    # Check if untouched data does not get updated

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(
                metadata={"team": "litellm-team3", "project": "litellm-project3"},
                team_id="litellm-core-infra@gmail.com",
            )
            key = await new_user(request)
            print(key)

            generated_key = key.key

            # use generated key to auth in
            result = await info_key_fn(key=generated_key)
            print("result from info_key_fn", result)
            assert result["key"] == generated_key
            print("\n info for key=", result["info"])
            assert result["info"]["max_parallel_requests"] == None
            assert result["info"]["metadata"] == {
                "team": "litellm-team3",
                "project": "litellm-project3",
            }
            assert result["info"]["team_id"] == "litellm-core-infra@gmail.com"

            request = Request(scope={"type": "http"})
            request._url = URL(url="/update/key")

            # update the key
            response1 = await update_key_fn(
                request=Request,
                data=UpdateKeyRequest(
                    key=generated_key,
                    models=["ada", "babbage", "curie", "davinci"],
                ),
            )

            print("response1=", response1)

            # update the team id
            response2 = await update_key_fn(
                request=Request,
                data=UpdateKeyRequest(key=generated_key, team_id="ishaan"),
            )
            print("response2=", response2)

            # get info on key after update
            result = await info_key_fn(key=generated_key)
            print("result from info_key_fn", result)
            assert result["key"] == generated_key
            print("\n info for key=", result["info"])
            assert result["info"]["max_parallel_requests"] == None
            assert result["info"]["metadata"] == {
                "team": "litellm-team3",
                "project": "litellm-project3",
            }
            assert result["info"]["models"] == ["ada", "babbage", "curie", "davinci"]
            assert result["info"]["team_id"] == "ishaan"

            # cleanup - delete key
            delete_key_request = KeyRequest(keys=[generated_key])

            # delete the key
            bearer_token = "Bearer sk-1234"

            request = Request(scope={"type": "http"})
            request._url = URL(url="/key/delete")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print(f"result: {result}")
            result.user_role = "proxy_admin"

            result_delete_key = await delete_key_fn(
                data=delete_key_request, user_api_key_dict=result
            )

        asyncio.run(test())
    except Exception as e:
        print("Got Exception", e)
        print(e.message)
        pytest.fail(f"An exception occurred - {str(e)}")


def test_key_generate_with_custom_auth(prisma_client):
    # custom - generate key function
    async def custom_generate_key_fn(data: GenerateKeyRequest) -> dict:
        """
        Asynchronous function for generating a key based on the input data.

        Args:
            data (GenerateKeyRequest): The input data for key generation.

        Returns:
            dict: A dictionary containing the decision and an optional message.
            {
                "decision": False,
                "message": "This violates LiteLLM Proxy Rules. No team id provided.",
            }
        """

        # decide if a key should be generated or not
        print("using custom auth function!")
        data_json = data.json()  # type: ignore

        # Unpacking variables
        team_id = data_json.get("team_id")
        duration = data_json.get("duration")
        models = data_json.get("models")
        aliases = data_json.get("aliases")
        config = data_json.get("config")
        spend = data_json.get("spend")
        user_id = data_json.get("user_id")
        max_parallel_requests = data_json.get("max_parallel_requests")
        metadata = data_json.get("metadata")
        tpm_limit = data_json.get("tpm_limit")
        rpm_limit = data_json.get("rpm_limit")

        if team_id is not None and team_id == "litellm-core-infra@gmail.com":
            # only team_id="litellm-core-infra@gmail.com" can make keys
            return {
                "decision": True,
            }
        else:
            print("Failed custom auth")
            return {
                "decision": False,
                "message": "This violates LiteLLM Proxy Rules. No team id provided.",
            }

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(
        litellm.proxy.proxy_server, "user_custom_key_generate", custom_generate_key_fn
    )
    try:

        async def test():
            try:
                await litellm.proxy.proxy_server.prisma_client.connect()
                request = GenerateKeyRequest()
                key = await generate_key_fn(request)
                pytest.fail(f"Expected an exception. Got {key}")
            except Exception as e:
                # this should fail
                print("Got Exception", e)
                print(e.message)
                print("First request failed!. This is expected")
                assert (
                    "This violates LiteLLM Proxy Rules. No team id provided."
                    in e.message
                )

            request_2 = GenerateKeyRequest(
                team_id="litellm-core-infra@gmail.com",
            )

            key = await generate_key_fn(request_2)
            print(key)
            generated_key = key.key

        asyncio.run(test())
    except Exception as e:
        print("Got Exception", e)
        print(e.message)
        pytest.fail(f"An exception occurred - {str(e)}")


def test_call_with_key_over_budget(prisma_client):
    # 12. Make a call with a key over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = GenerateKeyRequest(max_budget=0.00001)
            key = await generate_key_fn(request)
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
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )
            from litellm import ModelResponse, Choices, Message, Usage
            from litellm.caching import Cache

            litellm.cache = Cache()
            import time

            request_id = f"chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac{time.time()}"

            resp = ModelResponse(
                id=request_id,
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
                    "model": "chatgpt-v-2",
                    "stream": False,
                    "litellm_params": {
                        "metadata": {
                            "user_api_key": hash_token(generated_key),
                            "user_api_key_user_id": user_id,
                        }
                    },
                    "response_cost": 0.00002,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            await asyncio.sleep(10)
            # test spend_log was written and we can read it
            spend_logs = await view_spend_logs(request_id=request_id)

            print("read spend logs", spend_logs)
            assert len(spend_logs) == 1

            spend_log = spend_logs[0]

            assert spend_log.request_id == request_id
            assert spend_log.spend == float("2e-05")
            assert spend_log.model == "chatgpt-v-2"
            assert (
                spend_log.cache_key
                == "a61ae14fe4a8b8014a61e6ae01a100c8bc6770ac37c293242afed954bc69207d"
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        # print(f"Error - {str(e)}")
        traceback.print_exc()
        error_detail = e.message
        assert "Authentication Error, ExceededTokenBudget:" in error_detail
        print(vars(e))


def test_call_with_key_over_model_budget(prisma_client):
    # 12. Make a call with a key over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()

            # set budget for chatgpt-v-2 to 0.000001, expect the next request to fail
            request = GenerateKeyRequest(
                max_budget=1000,
                model_max_budget={
                    "chatgpt-v-2": 0.000001,
                },
                metadata={"user_api_key": 0.0001},
            )
            key = await generate_key_fn(request)
            print(key)

            generated_key = key.key
            user_id = key.user_id
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            async def return_body():
                return b'{"model": "chatgpt-v-2"}'

            request.body = return_body

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)

            # update spend using track_cost callback, make 2nd request, it should fail
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )
            from litellm import ModelResponse, Choices, Message, Usage
            from litellm.caching import Cache

            litellm.cache = Cache()
            import time

            request_id = f"chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac{time.time()}"

            resp = ModelResponse(
                id=request_id,
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
                    "model": "chatgpt-v-2",
                    "stream": False,
                    "litellm_params": {
                        "metadata": {
                            "user_api_key": hash_token(generated_key),
                            "user_api_key_user_id": user_id,
                        }
                    },
                    "response_cost": 0.00002,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            await asyncio.sleep(10)
            # test spend_log was written and we can read it
            spend_logs = await view_spend_logs(request_id=request_id)

            print("read spend logs", spend_logs)
            assert len(spend_logs) == 1

            spend_log = spend_logs[0]

            assert spend_log.request_id == request_id
            assert spend_log.spend == float("2e-05")
            assert spend_log.model == "chatgpt-v-2"
            assert (
                spend_log.cache_key
                == "a61ae14fe4a8b8014a61e6ae01a100c8bc6770ac37c293242afed954bc69207d"
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        # print(f"Error - {str(e)}")
        traceback.print_exc()
        error_detail = e.message
        assert "Authentication Error, ExceededModelBudget:" in error_detail
        print(vars(e))


@pytest.mark.asyncio()
async def test_call_with_key_never_over_budget(prisma_client):
    # Make a call with a key with budget=None, it should never fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:
        await litellm.proxy.proxy_server.prisma_client.connect()
        request = GenerateKeyRequest(max_budget=None)
        key = await generate_key_fn(request)
        print(key)

        generated_key = key.key
        user_id = key.user_id
        bearer_token = "Bearer " + generated_key

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        # use generated key to auth in
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        print("result from user auth with new key: {result}")

        # update spend using track_cost callback, make 2nd request, it should fail
        from litellm.proxy.proxy_server import (
            _PROXY_track_cost_callback as track_cost_callback,
        )
        from litellm import ModelResponse, Choices, Message, Usage
        import time

        request_id = f"chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac{time.time()}"

        resp = ModelResponse(
            id=request_id,
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
            usage=Usage(
                prompt_tokens=210000, completion_tokens=200000, total_tokens=41000
            ),
        )
        await track_cost_callback(
            kwargs={
                "model": "chatgpt-v-2",
                "stream": False,
                "litellm_params": {
                    "metadata": {
                        "user_api_key": hash_token(generated_key),
                        "user_api_key_user_id": user_id,
                    }
                },
                "response_cost": 200000,
            },
            completion_response=resp,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        await asyncio.sleep(5)
        # use generated key to auth in
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        print("result from user auth with new key", result)
    except Exception as e:
        pytest.fail(f"This should have not failed!. They key uses max_budget=None. {e}")


@pytest.mark.asyncio()
async def test_call_with_key_over_budget_stream(prisma_client):
    # 14. Make a call with a key over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    from litellm._logging import verbose_proxy_logger
    import logging

    litellm.set_verbose = True
    verbose_proxy_logger.setLevel(logging.DEBUG)
    try:
        await litellm.proxy.proxy_server.prisma_client.connect()
        request = GenerateKeyRequest(max_budget=0.00001)
        key = await generate_key_fn(request)
        print(key)

        generated_key = key.key
        user_id = key.user_id
        bearer_token = "Bearer " + generated_key
        print(f"generated_key: {generated_key}")
        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        # use generated key to auth in
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        print("result from user auth with new key", result)

        # update spend using track_cost callback, make 2nd request, it should fail
        from litellm.proxy.proxy_server import (
            _PROXY_track_cost_callback as track_cost_callback,
        )
        from litellm import ModelResponse, Choices, Message, Usage
        import time

        request_id = f"chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac{time.time()}"
        resp = ModelResponse(
            id=request_id,
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
                "call_type": "acompletion",
                "model": "sagemaker-chatgpt-v-2",
                "stream": True,
                "complete_streaming_response": resp,
                "litellm_params": {
                    "metadata": {
                        "user_api_key": hash_token(generated_key),
                        "user_api_key_user_id": user_id,
                    }
                },
                "response_cost": 0.00005,
            },
            completion_response=resp,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        await asyncio.sleep(5)
        # use generated key to auth in
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        print("result from user auth with new key", result)
        pytest.fail(f"This should have failed!. They key crossed it's budget")

    except Exception as e:
        print("Got Exception", e)
        error_detail = e.message
        assert "Authentication Error, ExceededTokenBudget:" in error_detail
        print(vars(e))


@pytest.mark.asyncio()
async def test_view_spend_per_user(prisma_client):
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        user_by_spend = await spend_user_fn(user_id=None)
        assert type(user_by_spend) == list
        assert len(user_by_spend) > 0
        first_user = user_by_spend[0]

        print("\nfirst_user=", first_user)
        assert first_user.spend > 0
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio()
async def test_view_spend_per_key(prisma_client):
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        key_by_spend = await spend_key_fn()
        assert type(key_by_spend) == list
        assert len(key_by_spend) > 0
        first_key = key_by_spend[0]

        print("\nfirst_key=", first_key)
        assert first_key.spend > 0
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio()
async def test_key_name_null(prisma_client):
    """
    - create key
    - get key info
    - assert key_name is null
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "general_settings", {"allow_user_auth": False})
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        request = GenerateKeyRequest()
        key = await generate_key_fn(request)
        generated_key = key.key
        result = await info_key_fn(key=generated_key)
        print("result from info_key_fn", result)
        assert result["info"]["key_name"] is None
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio()
async def test_key_name_set(prisma_client):
    """
    - create key
    - get key info
    - assert key_name is not null
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "general_settings", {"allow_user_auth": True})
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        request = GenerateKeyRequest()
        key = await generate_key_fn(request)
        generated_key = key.key
        result = await info_key_fn(key=generated_key)
        print("result from info_key_fn", result)
        assert isinstance(result["info"]["key_name"], str)
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio()
async def test_default_key_params(prisma_client):
    """
    - create key
    - get key info
    - assert key_name is not null
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "general_settings", {"allow_user_auth": True})
    litellm.default_key_generate_params = {"max_budget": 0.000122}
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        request = GenerateKeyRequest()
        key = await generate_key_fn(request)
        generated_key = key.key
        result = await info_key_fn(key=generated_key)
        print("result from info_key_fn", result)
        assert result["info"]["max_budget"] == 0.000122
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio()
async def test_upperbound_key_params(prisma_client):
    """
    - create key
    - get key info
    - assert key_name is not null
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    litellm.upperbound_key_generate_params = {
        "max_budget": 0.001,
        "budget_duration": "1m",
    }
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        request = GenerateKeyRequest(
            max_budget=200000,
            budget_duration="30d",
        )
        key = await generate_key_fn(request)
        generated_key = key.key

        result = await info_key_fn(key=generated_key)
        key_info = result["info"]
        # assert it used the upper bound for max_budget, and budget_duration
        assert key_info["max_budget"] == 0.001
        assert key_info["budget_duration"] == "1m"

        print(result)
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


def test_get_bearer_token():
    from litellm.proxy.proxy_server import _get_bearer_token

    # Test valid Bearer token
    api_key = "Bearer valid_token"
    result = _get_bearer_token(api_key)
    assert result == "valid_token", f"Expected 'valid_token', got '{result}'"

    # Test empty API key
    api_key = ""
    result = _get_bearer_token(api_key)
    assert result == "", f"Expected '', got '{result}'"

    # Test API key without Bearer prefix
    api_key = "invalid_token"
    result = _get_bearer_token(api_key)
    assert result == "", f"Expected '', got '{result}'"

    # Test API key with Bearer prefix in lowercase
    api_key = "bearer valid_token"
    result = _get_bearer_token(api_key)
    assert result == "", f"Expected '', got '{result}'"

    # Test API key with Bearer prefix and extra spaces
    api_key = "  Bearer   valid_token  "
    result = _get_bearer_token(api_key)
    assert result == "", f"Expected '', got '{result}'"

    # Test API key with Bearer prefix and no token
    api_key = "Bearer sk-1234"
    result = _get_bearer_token(api_key)
    assert result == "sk-1234", f"Expected 'valid_token', got '{result}'"


@pytest.mark.asyncio
async def test_user_api_key_auth(prisma_client):
    from litellm.proxy.proxy_server import ProxyException

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "general_settings", {"allow_user_auth": True})
    await litellm.proxy.proxy_server.prisma_client.connect()

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")
    # Test case: No API Key passed in
    try:
        await user_api_key_auth(request, api_key=None)
        pytest.fail(f"This should have failed!. IT's an invalid key")
    except ProxyException as exc:
        print(exc.message)
        assert exc.message == "Authentication Error, No api key passed in."

    # Test case: Malformed API Key (missing 'Bearer ' prefix)
    try:
        await user_api_key_auth(request, api_key="my_token")
        pytest.fail(f"This should have failed!. IT's an invalid key")
    except ProxyException as exc:
        print(exc.message)
        assert (
            exc.message
            == "Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix. Passed in: my_token"
        )

    # Test case: User passes empty string API Key
    try:
        await user_api_key_auth(request, api_key="")
        pytest.fail(f"This should have failed!. IT's an invalid key")
    except ProxyException as exc:
        print(exc.message)
        assert (
            exc.message
            == "Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix. Passed in: "
        )


@pytest.mark.asyncio
async def test_user_api_key_auth_without_master_key(prisma_client):
    # if master key is not set, expect all calls to go through
    try:
        from litellm.proxy.proxy_server import ProxyException

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", None)
        setattr(
            litellm.proxy.proxy_server, "general_settings", {"allow_user_auth": True}
        )
        await litellm.proxy.proxy_server.prisma_client.connect()

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")
        # Test case: No API Key passed in

        await user_api_key_auth(request, api_key=None)
        await user_api_key_auth(request, api_key="my_token")
        await user_api_key_auth(request, api_key="")
        await user_api_key_auth(request, api_key="Bearer " + "1234")
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio
async def test_key_with_no_permissions(prisma_client):
    """
    - create key
    - get key info
    - assert key_name is null
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "general_settings", {"allow_user_auth": False})
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        response = await generate_key_helper_fn(
            **{"duration": "1hr", "key_max_budget": 0, "models": [], "aliases": {}, "config": {}, "spend": 0, "user_id": "ishaan", "team_id": "litellm-dashboard"}  # type: ignore
        )

        print(response)
        key = response["token"]

        # make a /chat/completions call -> it should fail
        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        # use generated key to auth in
        result = await user_api_key_auth(request=request, api_key="Bearer " + key)
        print("result from user auth with new key", result)
        pytest.fail(f"This should have failed!. IT's an invalid key")
    except Exception as e:
        print("Got Exception", e)
        print(e.message)


async def track_cost_callback_helper_fn(generated_key: str, user_id: str):
    from litellm import ModelResponse, Choices, Message, Usage
    from litellm.proxy.proxy_server import (
        _PROXY_track_cost_callback as track_cost_callback,
    )

    import uuid

    request_id = f"chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac{uuid.uuid4()}"
    resp = ModelResponse(
        id=request_id,
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
            "call_type": "acompletion",
            "model": "sagemaker-chatgpt-v-2",
            "stream": True,
            "complete_streaming_response": resp,
            "litellm_params": {
                "metadata": {
                    "user_api_key": hash_token(generated_key),
                    "user_api_key_user_id": user_id,
                }
            },
            "response_cost": 0.00005,
        },
        completion_response=resp,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )


@pytest.mark.skip(reason="High traffic load test for spend tracking")
@pytest.mark.asyncio
async def test_proxy_load_test_db(prisma_client):
    """
    Run 1500 req./s against track_cost_callback function
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    from litellm._logging import verbose_proxy_logger
    import logging, time

    litellm.set_verbose = True
    verbose_proxy_logger.setLevel(logging.DEBUG)
    try:
        start_time = time.time()
        await litellm.proxy.proxy_server.prisma_client.connect()
        request = GenerateKeyRequest(max_budget=0.00001)
        key = await generate_key_fn(request)
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
        n = 5000
        tasks = [
            track_cost_callback_helper_fn(generated_key=generated_key, user_id=user_id)
            for _ in range(n)
        ]
        completions = await asyncio.gather(*tasks)
        await asyncio.sleep(120)
        try:
            # call spend logs
            spend_logs = await view_spend_logs(api_key=generated_key)

            print(f"len responses: {len(spend_logs)}")
            assert len(spend_logs) == n
            print(n, time.time() - start_time, len(spend_logs))
        except:
            print(n, time.time() - start_time, 0)
        raise Exception(f"it worked! key={key.key}")
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")
