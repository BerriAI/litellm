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


# function to call to generate key - async def new_user(data: NewUserRequest):
# function to validate a request - async def user_auth(request: Request):

import sys, os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

load_dotenv()
import os, io

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
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm._logging import verbose_proxy_logger

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from litellm.proxy._types import (
    NewUserRequest,
    GenerateKeyRequest,
    DynamoDBArgs,
    DeleteKeyRequest,
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
                    "response_cost": 0.00002,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
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
                    "response_cost": 0.00002,
                },
                completion_response=ModelResponse(),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

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
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest()
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            delete_key_request = DeleteKeyRequest(keys=[generated_key])

            # delete the key
            result_delete_key = await delete_key_fn(data=delete_key_request)
            print("result from delete key", result_delete_key)
            assert result_delete_key == {"deleted_keys": [generated_key]}

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
            request = NewUserRequest()
            key = await new_user(request)
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            delete_key_request = DeleteKeyRequest(keys=[generated_key])

            # delete the key
            result_delete_key = await delete_key_fn(data=delete_key_request)

            print("result from delete key", result_delete_key)
            assert result_delete_key == {"deleted_keys": [generated_key]}

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
            delete_key_request = DeleteKeyRequest(keys=[generated_key])

            # delete the key
            await delete_key_fn(data=delete_key_request)

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
            await update_key_fn(
                request=Request,
                data=UpdateKeyRequest(
                    key=generated_key,
                    models=["ada", "babbage", "curie", "davinci"],
                ),
            )

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

            # cleanup - delete key
            delete_key_request = DeleteKeyRequest(keys=[generated_key])

            # delete the key
            await delete_key_fn(data=delete_key_request)

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
                    "response_cost": 0.00002,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
        assert "Authentication Error, ExceededTokenBudget:" in error_detail
        print(vars(e))


def test_call_with_key_over_budget_stream(prisma_client):
    # 14. Make a call with a key over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    from litellm._logging import verbose_proxy_logger
    import logging

    litellm.set_verbose = True
    verbose_proxy_logger.setLevel(logging.DEBUG)
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
                    "response_cost": 0.00002,
                },
                completion_response=ModelResponse(),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

    except Exception as e:
        error_detail = e.message
        assert "Authentication Error, ExceededTokenBudget:" in error_detail
        print(vars(e))
