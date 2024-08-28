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

import os
import sys
import traceback
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import io
import os
import time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    new_user,
    user_info,
    user_update,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_key_fn,
    generate_key_fn,
    generate_key_helper_fn,
    info_key_fn,
    regenerate_key_fn,
    update_key_fn,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    new_team,
    team_info,
    update_team,
)
from litellm.proxy.proxy_server import (
    LitellmUserRoles,
    audio_transcriptions,
    chat_completion,
    completion,
    embeddings,
    image_generation,
    model_list,
    moderations,
    new_end_user,
    user_api_key_auth,
)
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    global_spend,
    spend_key_fn,
    spend_user_fn,
    view_spend_logs,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from starlette.datastructures import URL

from litellm.caching import DualCache
from litellm.proxy._types import (
    DynamoDBArgs,
    GenerateKeyRequest,
    KeyRequest,
    LiteLLM_UpperboundKeyGenerateParams,
    NewCustomerRequest,
    NewTeamRequest,
    NewUserRequest,
    ProxyErrorTypes,
    ProxyException,
    UpdateKeyRequest,
    UpdateTeamRequest,
    UpdateUserRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import DBClient

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


@pytest.mark.asyncio()
async def test_new_user_response(prisma_client):
    try:

        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

        await litellm.proxy.proxy_server.prisma_client.connect()
        from litellm.proxy.proxy_server import user_api_key_cache

        _team_id = "ishaan-special-team_{}".format(uuid.uuid4())
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )

        _response = await new_user(
            data=NewUserRequest(
                models=["azure-gpt-3.5"],
                team_id=_team_id,
                tpm_limit=20,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
        )
        print(_response)
        assert _response.models == ["azure-gpt-3.5"]
        assert _response.team_id == _team_id
        assert _response.tpm_limit == 20

    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.parametrize(
    "api_route",
    [
        # chat_completion
        APIRoute(path="/engines/{model}/chat/completions", endpoint=chat_completion),
        APIRoute(
            path="/openai/deployments/{model}/chat/completions",
            endpoint=chat_completion,
        ),
        APIRoute(path="/chat/completions", endpoint=chat_completion),
        APIRoute(path="/v1/chat/completions", endpoint=chat_completion),
        # completion
        APIRoute(path="/completions", endpoint=completion),
        APIRoute(path="/v1/completions", endpoint=completion),
        APIRoute(path="/engines/{model}/completions", endpoint=completion),
        APIRoute(path="/openai/deployments/{model}/completions", endpoint=completion),
        # embeddings
        APIRoute(path="/v1/embeddings", endpoint=embeddings),
        APIRoute(path="/embeddings", endpoint=embeddings),
        APIRoute(path="/openai/deployments/{model}/embeddings", endpoint=embeddings),
        # image generation
        APIRoute(path="/v1/images/generations", endpoint=image_generation),
        APIRoute(path="/images/generations", endpoint=image_generation),
        # audio transcriptions
        APIRoute(path="/v1/audio/transcriptions", endpoint=audio_transcriptions),
        APIRoute(path="/audio/transcriptions", endpoint=audio_transcriptions),
        # moderations
        APIRoute(path="/v1/moderations", endpoint=moderations),
        APIRoute(path="/moderations", endpoint=moderations),
        # model_list
        APIRoute(path="/v1/models", endpoint=model_list),
        APIRoute(path="/models", endpoint=model_list),
        # threads
        APIRoute(
            path="/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ", endpoint=model_list
        ),
    ],
    ids=lambda route: str(dict(route=route.endpoint.__name__, path=route.path)),
)
def test_generate_and_call_with_valid_key(prisma_client, api_route):
    # 1. Generate a Key, and use it to make a call

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            from litellm.proxy.proxy_server import user_api_key_cache

            request = NewUserRequest(user_role=LitellmUserRoles.INTERNAL_USER)
            key = await new_user(
                request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
            print(key)
            user_id = key.user_id

            # check /user/info to verify user_role was set correctly
            new_user_info = await user_info(user_id=user_id)
            new_user_info = new_user_info["user_info"]
            print("new_user_info=", new_user_info)
            assert new_user_info.user_role == LitellmUserRoles.INTERNAL_USER
            assert new_user_info.user_id == user_id

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            assert generated_key not in user_api_key_cache.in_memory_cache.cache_dict

            value_from_prisma = await prisma_client.get_data(
                token=generated_key,
            )
            print("token from prisma", value_from_prisma)

            request = Request(
                {
                    "type": "http",
                    "route": api_route,
                    "path": api_route.path,
                    "headers": [("Authorization", bearer_token)],
                }
            )

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
            generated_key = "sk-126666"
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
        assert "Authentication Error, Invalid proxy server token passed" in e.message
        pass


def test_call_with_invalid_model(prisma_client):
    litellm.set_verbose = True
    # 3. Make a call to a key with an invalid model - expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(models=["mistral"])
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
            print(key)

            generated_key = key.key
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            async def return_body():
                return b'{"model": "gemini-pro-vision"}'

            request.body = return_body

            # use generated key to auth in
            print(
                "Bearer token being sent to user_api_key_auth() - {}".format(
                    bearer_token
                )
            )
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
            key = await new_user(
                request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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


@pytest.mark.asyncio
async def test_call_with_valid_model_using_all_models(prisma_client):
    """
    Do not delete
    this is the Admin UI flow
    1. Create a team with model = `all-proxy-models`
    2. Create a key with model = `all-team-models`
    3. Call /chat/completions with the key -> expect to pass
    """
    # Make a call to a key with model = `all-proxy-models` this is an Alias from LiteLLM Admin UI
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        await litellm.proxy.proxy_server.prisma_client.connect()

        team_request = NewTeamRequest(
            team_alias="testing-team",
            models=["all-proxy-models"],
        )

        new_team_response = await new_team(
            data=team_request,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
            http_request=Request(scope={"type": "http"}),
        )
        print("new_team_response", new_team_response)
        created_team_id = new_team_response["team_id"]

        request = GenerateKeyRequest(
            models=["all-team-models"], team_id=created_team_id
        )
        key = await generate_key_fn(data=request)
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

        # call /key/info for key - models == "all-proxy-models"
        key_info = await info_key_fn(key=generated_key)
        print("key_info", key_info)
        models = key_info["info"]["models"]
        assert models == ["all-team-models"]

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
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )

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
        assert "ExceededBudget:" in error_detail
        assert isinstance(e, ProxyException)
        assert e.type == ProxyErrorTypes.budget_exceeded
        print(vars(e))


def test_end_user_cache_write_unit_test():
    """
    assert end user object is being written to cache as expected
    """
    pass


def test_call_with_end_user_over_budget(prisma_client):
    # Test if a user passed to /chat/completions is tracked & fails when they cross their budget
    # we only check this when litellm.max_end_user_budget is set
    import random

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm, "max_end_user_budget", 0.00001)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            user = f"ishaan {uuid.uuid4().hex}"
            request = NewCustomerRequest(
                user_id=user, max_budget=0.000001
            )  # create a key with no budget
            await new_end_user(
                request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")
            bearer_token = "Bearer sk-1234"

            result = await user_api_key_auth(request=request, api_key=bearer_token)

            async def return_body():
                return_string = f'{{"model": "gemini-pro-vision", "user": "{user}"}}'
                # return string as bytes
                return return_string.encode()

            request.body = return_body

            # update spend using track_cost callback, make 2nd request, it should fail
            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )

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
                            "user_api_key": "sk-1234",
                            "user_api_key_user_id": user,
                        },
                        "proxy_server_request": {
                            "body": {
                                "user": user,
                            }
                        },
                    },
                    "response_cost": 10,
                },
                completion_response=resp,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

            await asyncio.sleep(10)
            await update_spend(
                prisma_client=prisma_client,
                db_writer_client=None,
                proxy_logging_obj=proxy_logging_obj,
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail("This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
        assert "Budget has been exceeded! Current" in error_detail
        assert isinstance(e, ProxyException)
        assert e.type == ProxyErrorTypes.budget_exceeded
        print(vars(e))


def test_call_with_proxy_over_budget(prisma_client):
    # 5.1 Make a call with a proxy over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    litellm_proxy_budget_name = f"litellm-proxy-budget-{time.time()}"
    setattr(
        litellm.proxy.proxy_server,
        "litellm_proxy_admin_name",
        litellm_proxy_budget_name,
    )
    setattr(litellm, "max_budget", 0.00001)
    from litellm.proxy.proxy_server import user_api_key_cache

    user_api_key_cache.set_cache(
        key="{}:spend".format(litellm_proxy_budget_name), value=0
    )
    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest()
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )

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
        assert "Budget has been exceeded" in error_detail
        assert isinstance(e, ProxyException)
        assert e.type == ProxyErrorTypes.budget_exceeded
        print(vars(e))


def test_call_with_user_over_budget_stream(prisma_client):
    # 6. Make a call with a key over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    import logging

    from litellm._logging import verbose_proxy_logger

    litellm.set_verbose = True
    verbose_proxy_logger.setLevel(logging.DEBUG)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            request = NewUserRequest(max_budget=0.00001)
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )

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
            pytest.fail("This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        error_detail = e.message
        assert "ExceededBudget:" in error_detail
        assert isinstance(e, ProxyException)
        assert e.type == ProxyErrorTypes.budget_exceeded
        print(vars(e))


def test_call_with_proxy_over_budget_stream(prisma_client):
    # 6.1 Make a call with a global proxy over budget, expect to fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    litellm_proxy_budget_name = f"litellm-proxy-budget-{time.time()}"
    setattr(
        litellm.proxy.proxy_server,
        "litellm_proxy_admin_name",
        litellm_proxy_budget_name,
    )
    setattr(litellm, "max_budget", 0.00001)
    from litellm.proxy.proxy_server import user_api_key_cache

    user_api_key_cache.set_cache(
        key="{}:spend".format(litellm_proxy_budget_name), value=0
    )
    setattr(litellm.proxy.proxy_server, "user_api_key_cache", user_api_key_cache)

    import logging

    from litellm._logging import verbose_proxy_logger

    litellm.set_verbose = True
    verbose_proxy_logger.setLevel(logging.DEBUG)
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()
            ## CREATE PROXY + USER BUDGET ##
            # request = NewUserRequest(
            #     max_budget=0.00001, user_id=litellm_proxy_budget_name
            # )
            request = NewUserRequest()
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )

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
        assert "Budget has been exceeded" in error_detail
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
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
        assert e.type == ProxyErrorTypes.expired_key

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
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            result.user_role = LitellmUserRoles.PROXY_ADMIN
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
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            result.user_role = LitellmUserRoles.PROXY_ADMIN

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
            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
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
            result.user_role = LitellmUserRoles.PROXY_ADMIN

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
    import uuid

    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            await litellm.proxy.proxy_server.prisma_client.connect()

            # create team "litellm-core-infra@gmail.com""
            print("creating team litellm-core-infra@gmail.com")
            _team_1 = "litellm-core-infra@gmail.com_{}".format(uuid.uuid4())
            await new_team(
                NewTeamRequest(
                    team_id=_team_1,
                ),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
                http_request=Request(scope={"type": "http"}),
            )

            _team_2 = "ishaan-special-team_{}".format(uuid.uuid4())
            await new_team(
                NewTeamRequest(
                    team_id=_team_2,
                ),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
                http_request=Request(scope={"type": "http"}),
            )

            request = NewUserRequest(
                metadata={"project": "litellm-project3"},
                team_id=_team_1,
            )

            key = await new_user(
                data=request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key="sk-1234",
                    user_id="1234",
                ),
            )
            print(key)

            generated_key = key.key

            # use generated key to auth in
            result = await info_key_fn(key=generated_key)
            print("result from info_key_fn", result)
            assert result["key"] == generated_key
            print("\n info for key=", result["info"])
            assert result["info"]["max_parallel_requests"] == None
            assert result["info"]["metadata"] == {
                "project": "litellm-project3",
            }
            assert result["info"]["team_id"] == _team_1

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
                data=UpdateKeyRequest(key=generated_key, team_id=_team_2),
            )
            print("response2=", response2)

            # get info on key after update
            result = await info_key_fn(key=generated_key)
            print("result from info_key_fn", result)
            assert result["key"] == generated_key
            print("\n info for key=", result["info"])
            assert result["info"]["max_parallel_requests"] == None
            assert result["info"]["metadata"] == {
                "project": "litellm-project3",
            }
            assert result["info"]["models"] == ["ada", "babbage", "curie", "davinci"]
            assert result["info"]["team_id"] == _team_2

            # cleanup - delete key
            delete_key_request = KeyRequest(keys=[generated_key])

            # delete the key
            bearer_token = "Bearer sk-1234"

            request = Request(scope={"type": "http"})
            request._url = URL(url="/key/delete")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print(f"result: {result}")
            result.user_role = LitellmUserRoles.PROXY_ADMIN

            result_delete_key = await delete_key_fn(
                data=delete_key_request, user_api_key_dict=result
            )

        asyncio.run(test())
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"An exception occurred - {str(e)}\n{traceback.format_exc()}")


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
            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.caching import Cache
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )

            litellm.cache = Cache()
            import time
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
            await update_spend(
                prisma_client=prisma_client,
                db_writer_client=None,
                proxy_logging_obj=proxy_logging_obj,
            )
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
                == "c891d64397a472e6deb31b87a5ac4d3ed5b2dcc069bc87e2afe91e6d64e95a1e"
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail("This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        # print(f"Error - {str(e)}")
        traceback.print_exc()
        if hasattr(e, "message"):
            error_detail = e.message
        else:
            error_detail = str(e)
        assert "Budget has been exceeded" in error_detail
        assert isinstance(e, ProxyException)
        assert e.type == ProxyErrorTypes.budget_exceeded
        print(vars(e))


def test_call_with_key_over_budget_no_cache(prisma_client):
    # 12. Make a call with a key over budget, expect to fail
    # ✅  Tests if spend trackign works when the key does not exist in memory
    # Related to this: https://github.com/BerriAI/litellm/issues/3920
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
            from litellm.proxy.proxy_server import user_api_key_cache

            user_api_key_cache.in_memory_cache.cache_dict = {}
            setattr(litellm.proxy.proxy_server, "proxy_batch_write_at", 1)

            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.caching import Cache

            litellm.cache = Cache()
            import time
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
            await update_spend(
                prisma_client=prisma_client,
                db_writer_client=None,
                proxy_logging_obj=proxy_logging_obj,
            )
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
                == "c891d64397a472e6deb31b87a5ac4d3ed5b2dcc069bc87e2afe91e6d64e95a1e"
            )

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            print("result from user auth with new key", result)
            pytest.fail(f"This should have failed!. They key crossed it's budget")

        asyncio.run(test())
    except Exception as e:
        # print(f"Error - {str(e)}")
        traceback.print_exc()
        if hasattr(e, "message"):
            error_detail = e.message
        else:
            error_detail = str(e)
        assert "Budget has been exceeded" in error_detail
        assert isinstance(e, ProxyException)
        assert e.type == ProxyErrorTypes.budget_exceeded
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
            from litellm import Choices, Message, ModelResponse, Usage
            from litellm.caching import Cache
            from litellm.proxy.proxy_server import (
                _PROXY_track_cost_callback as track_cost_callback,
            )

            litellm.cache = Cache()
            import time
            import uuid

            request_id = f"chatcmpl-{uuid.uuid4()}"

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
            await update_spend(
                prisma_client=prisma_client,
                db_writer_client=None,
                proxy_logging_obj=proxy_logging_obj,
            )
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
                == "c891d64397a472e6deb31b87a5ac4d3ed5b2dcc069bc87e2afe91e6d64e95a1e"
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
        assert "Budget has been exceeded!" in error_detail
        assert isinstance(e, ProxyException)
        assert e.type == ProxyErrorTypes.budget_exceeded
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
        import time
        import uuid

        from litellm import Choices, Message, ModelResponse, Usage
        from litellm.proxy.proxy_server import (
            _PROXY_track_cost_callback as track_cost_callback,
        )

        request_id = f"chatcmpl-{uuid.uuid4()}"

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
        await update_spend(
            prisma_client=prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )
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
    import logging

    from litellm._logging import verbose_proxy_logger

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
        import time
        import uuid

        from litellm import Choices, Message, ModelResponse, Usage
        from litellm.proxy.proxy_server import (
            _PROXY_track_cost_callback as track_cost_callback,
        )

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
        await update_spend(
            prisma_client=prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )
        # use generated key to auth in
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        print("result from user auth with new key", result)
        pytest.fail(f"This should have failed!. They key crossed it's budget")

    except Exception as e:
        print("Got Exception", e)
        error_detail = e.message
        assert "Budget has been exceeded" in error_detail

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
        assert first_user["spend"] > 0
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
    os.environ["DISABLE_KEY_NAME"] = "True"
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        request = GenerateKeyRequest()
        key = await generate_key_fn(request)
        print("generated key=", key)
        generated_key = key.key
        result = await info_key_fn(key=generated_key)
        print("result from info_key_fn", result)
        assert result["info"]["key_name"] is None
    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")
    finally:
        os.environ["DISABLE_KEY_NAME"] = "False"


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
    litellm.upperbound_key_generate_params = LiteLLM_UpperboundKeyGenerateParams(
        max_budget=0.001, budget_duration="1m"
    )
    await litellm.proxy.proxy_server.prisma_client.connect()
    try:
        request = GenerateKeyRequest(
            max_budget=200000,
            budget_duration="30d",
        )
        key = await generate_key_fn(request)
        # print(result)
    except Exception as e:
        assert e.code == str(400)


def test_get_bearer_token():
    from litellm.proxy.auth.user_api_key_auth import _get_bearer_token

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

    # Test API key with Bearer prefix and extra spaces
    api_key = "  Bearer   valid_token  "
    result = _get_bearer_token(api_key)
    assert result == "", f"Expected '', got '{result}'"

    # Test API key with Bearer prefix and no token
    api_key = "Bearer sk-1234"
    result = _get_bearer_token(api_key)
    assert result == "sk-1234", f"Expected 'valid_token', got '{result}'"


def test_update_logs_with_spend_logs_url(prisma_client):
    """
    Unit test for making sure spend logs list is still updated when url passed in
    """
    from litellm.proxy.proxy_server import _set_spend_logs_payload

    payload = {"startTime": datetime.now(), "endTime": datetime.now()}
    _set_spend_logs_payload(payload=payload, prisma_client=prisma_client)

    assert len(prisma_client.spend_log_transactions) > 0

    prisma_client.spend_log_transactions = []

    spend_logs_url = ""
    payload = {"startTime": datetime.now(), "endTime": datetime.now()}
    _set_spend_logs_payload(
        payload=payload, spend_logs_url=spend_logs_url, prisma_client=prisma_client
    )

    assert len(prisma_client.spend_log_transactions) > 0


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
            request_type="key",
            **{"duration": "1hr", "key_max_budget": 0, "models": [], "aliases": {}, "config": {}, "spend": 0, "user_id": "ishaan", "team_id": "litellm-dashboard"},  # type: ignore
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
    import uuid

    from litellm import Choices, Message, ModelResponse, Usage
    from litellm.proxy.proxy_server import (
        _PROXY_track_cost_callback as track_cost_callback,
    )

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
    import logging
    import time

    from litellm._logging import verbose_proxy_logger

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


@pytest.mark.asyncio()
async def test_master_key_hashing(prisma_client):
    try:
        import uuid

        print("prisma client=", prisma_client)

        master_key = "sk-1234"

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", master_key)

        await litellm.proxy.proxy_server.prisma_client.connect()
        from litellm.proxy.proxy_server import user_api_key_cache

        _team_id = "ishaans-special-team_{}".format(uuid.uuid4())
        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        )
        await new_team(
            NewTeamRequest(team_id=_team_id),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
            http_request=Request(scope={"type": "http"}),
        )

        _response = await new_user(
            data=NewUserRequest(
                models=["azure-gpt-3.5"],
                team_id=_team_id,
                tpm_limit=20,
            ),
            user_api_key_dict=user_api_key_dict,
        )
        print(_response)
        assert _response.models == ["azure-gpt-3.5"]
        assert _response.team_id == _team_id
        assert _response.tpm_limit == 20

        bearer_token = "Bearer " + master_key

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        # use generated key to auth in
        result: UserAPIKeyAuth = await user_api_key_auth(
            request=request, api_key=bearer_token
        )

        assert result.api_key == hash_token(master_key)

    except Exception as e:
        print("Got Exception", e)
        pytest.fail(f"Got exception {e}")


@pytest.mark.asyncio
async def test_reset_spend_authentication(prisma_client):
    """
    1. Test master key can access this route  -> ONLY MASTER KEY SHOULD BE ABLE TO RESET SPEND
    2. Test that non-master key gets rejected
    3. Test that non-master key with role == LitellmUserRoles.PROXY_ADMIN or admin gets rejected
    """

    print("prisma client=", prisma_client)

    master_key = "sk-1234"

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)

    await litellm.proxy.proxy_server.prisma_client.connect()
    from litellm.proxy.proxy_server import user_api_key_cache

    bearer_token = "Bearer " + master_key

    request = Request(scope={"type": "http"})
    request._url = URL(url="/global/spend/reset")

    # Test 1 - Master Key
    result: UserAPIKeyAuth = await user_api_key_auth(
        request=request, api_key=bearer_token
    )

    print("result from user auth with Master key", result)
    assert result.token is not None

    # Test 2 - Non-Master Key
    _response = await new_user(
        data=NewUserRequest(
            tpm_limit=20,
        )
    )

    generate_key = "Bearer " + _response.key

    try:
        await user_api_key_auth(request=request, api_key=generate_key)
        pytest.fail(f"This should have failed!. IT's an expired key")
    except Exception as e:
        print("Got Exception", e)
        assert (
            "Tried to access route=/global/spend/reset, which is only for MASTER KEY"
            in e.message
        )

    # Test 3 - Non-Master Key with role == LitellmUserRoles.PROXY_ADMIN or admin
    _response = await new_user(
        data=NewUserRequest(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            tpm_limit=20,
        )
    )

    generate_key = "Bearer " + _response.key

    try:
        await user_api_key_auth(request=request, api_key=generate_key)
        pytest.fail(f"This should have failed!. IT's an expired key")
    except Exception as e:
        print("Got Exception", e)
        assert (
            "Tried to access route=/global/spend/reset, which is only for MASTER KEY"
            in e.message
        )


@pytest.mark.asyncio()
async def test_create_update_team(prisma_client):
    """
    - Set max_budget, budget_duration, max_budget, tpm_limit, rpm_limit
    - Assert response has correct values

    - Update max_budget, budget_duration, max_budget, tpm_limit, rpm_limit
    - Assert response has correct values

    - Call team_info and assert response has correct values
    """
    print("prisma client=", prisma_client)

    master_key = "sk-1234"

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    import datetime

    await litellm.proxy.proxy_server.prisma_client.connect()
    from litellm.proxy.proxy_server import user_api_key_cache

    _team_id = "test-team_{}".format(uuid.uuid4())
    response = await new_team(
        NewTeamRequest(
            team_id=_team_id,
            max_budget=20,
            budget_duration="30d",
            tpm_limit=20,
            rpm_limit=20,
        ),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    print("RESPONSE from new_team", response)

    assert response["team_id"] == _team_id
    assert response["max_budget"] == 20
    assert response["tpm_limit"] == 20
    assert response["rpm_limit"] == 20
    assert response["budget_duration"] == "30d"
    assert response["budget_reset_at"] is not None and isinstance(
        response["budget_reset_at"], datetime.datetime
    )

    # updating team budget duration and reset at

    response = await update_team(
        UpdateTeamRequest(
            team_id=_team_id,
            max_budget=30,
            budget_duration="2d",
            tpm_limit=30,
            rpm_limit=30,
        ),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    print("RESPONSE from update_team", response)
    _updated_info = response["data"]
    _updated_info = dict(_updated_info)

    assert _updated_info["team_id"] == _team_id
    assert _updated_info["max_budget"] == 30
    assert _updated_info["tpm_limit"] == 30
    assert _updated_info["rpm_limit"] == 30
    assert _updated_info["budget_duration"] == "2d"
    assert _updated_info["budget_reset_at"] is not None and isinstance(
        _updated_info["budget_reset_at"], datetime.datetime
    )

    # now hit team_info
    try:
        response = await team_info(
            team_id=_team_id,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
        )
    except Exception as e:
        print(e)
        pytest.fail("Receives error - {}".format(e))

    _team_info = response["team_info"]
    _team_info = dict(_team_info)

    assert _team_info["team_id"] == _team_id
    assert _team_info["max_budget"] == 30
    assert _team_info["tpm_limit"] == 30
    assert _team_info["rpm_limit"] == 30
    assert _team_info["budget_duration"] == "2d"
    assert _team_info["budget_reset_at"] is not None and isinstance(
        _team_info["budget_reset_at"], datetime.datetime
    )


@pytest.mark.asyncio()
async def test_enforced_params(prisma_client):
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    from litellm.proxy.proxy_server import general_settings

    general_settings["enforced_params"] = [
        "user",
        "metadata",
        "metadata.generation_name",
    ]

    await litellm.proxy.proxy_server.prisma_client.connect()
    request = NewUserRequest()
    key = await new_user(
        data=request,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print(key)

    generated_key = key.key
    bearer_token = "Bearer " + generated_key

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    # Case 1: Missing user
    async def return_body():
        return b'{"model": "gemini-pro-vision"}'

    request.body = return_body
    try:
        await user_api_key_auth(request=request, api_key=bearer_token)
        pytest.fail(f"This should have failed!. IT's an invalid request")
    except Exception as e:
        assert (
            "BadRequest please pass param=user in request body. This is a required param"
            in e.message
        )

    # Case 2: Missing metadata["generation_name"]
    async def return_body_2():
        return b'{"model": "gemini-pro-vision", "user": "1234", "metadata": {}}'

    request.body = return_body_2
    try:
        await user_api_key_auth(request=request, api_key=bearer_token)
        pytest.fail(f"This should have failed!. IT's an invalid request")
    except Exception as e:
        assert (
            "Authentication Error, BadRequest please pass param=[metadata][generation_name] in request body"
            in e.message
        )
    general_settings.pop("enforced_params")


@pytest.mark.asyncio()
async def test_update_user_role(prisma_client):
    """
    Tests if we update user role, incorrect values are not stored in cache
    -> create a user with role == INTERNAL_USER
    -> access an Admin only route -> expect to fail
    -> update user role to == PROXY_ADMIN
    -> access an Admin only route -> expect to succeed
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    key = await new_user(
        data=NewUserRequest(
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
    )

    print(key)
    api_key = "Bearer " + key.key

    api_route = APIRoute(path="/global/spend", endpoint=global_spend)
    request = Request(
        {
            "type": "http",
            "route": api_route,
            "path": "/global/spend",
            "headers": [("Authorization", api_key)],
        }
    )

    request._url = URL(url="/global/spend")

    # use generated key to auth in
    try:
        result = await user_api_key_auth(request=request, api_key=api_key)
        print("result from user auth with new key", result)
    except Exception as e:
        print(e)
        pass

    await user_update(
        data=UpdateUserRequest(
            user_id=key.user_id, user_role=LitellmUserRoles.PROXY_ADMIN
        )
    )

    await asyncio.sleep(2)

    # use generated key to auth in
    print("\n\nMAKING NEW REQUEST WITH UPDATED USER ROLE\n\n")
    result = await user_api_key_auth(request=request, api_key=api_key)
    print("result from user auth with new key", result)


@pytest.mark.asyncio()
async def test_custom_api_key_header_name(prisma_client):
    """ """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(
        litellm.proxy.proxy_server,
        "general_settings",
        {"litellm_key_header_name": "x-litellm-key"},
    )
    await litellm.proxy.proxy_server.prisma_client.connect()

    api_route = APIRoute(path="/chat/completions", endpoint=chat_completion)
    request = Request(
        {
            "type": "http",
            "route": api_route,
            "path": api_route.path,
            "headers": [
                (b"x-litellm-key", b"Bearer sk-1234"),
            ],
        }
    )

    # this should pass because we pass the master key as X-Litellm-Key and litellm_key_header_name="X-Litellm-Key" in general settings
    result = await user_api_key_auth(request=request, api_key="Bearer invalid-key")

    # this should fail because X-Litellm-Key is invalid
    request = Request(
        {
            "type": "http",
            "route": api_route,
            "path": api_route.path,
            "headers": [],
        }
    )
    try:
        result = await user_api_key_auth(request=request, api_key="Bearer sk-1234")
        pytest.fail(f"This should have failed!. invalid Auth on this request")
    except Exception as e:
        print("failed with error", e)
        assert (
            "No LiteLLM Virtual Key pass. Please set header=x-litellm-key: Bearer <api_key>"
            in e.message
        )
        pass

    # this should pass because X-Litellm-Key is valid


@pytest.mark.asyncio()
async def test_generate_key_with_model_tpm_limit(prisma_client):
    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    request = GenerateKeyRequest(
        metadata={
            "team": "litellm-team3",
            "model_tpm_limit": {"gpt-4": 100},
            "model_rpm_limit": {"gpt-4": 2},
        }
    )
    key = await generate_key_fn(
        data=request,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print(key)

    generated_key = key.key

    # use generated key to auth in
    result = await info_key_fn(key=generated_key)
    print("result from info_key_fn", result)
    assert result["key"] == generated_key
    print("\n info for key=", result["info"])
    assert result["info"]["metadata"] == {
        "team": "litellm-team3",
        "model_tpm_limit": {"gpt-4": 100},
        "model_rpm_limit": {"gpt-4": 2},
    }

    # Update model tpm_limit and rpm_limit
    request = UpdateKeyRequest(
        key=generated_key,
        model_tpm_limit={"gpt-4": 200},
        model_rpm_limit={"gpt-4": 3},
    )
    _request = Request(scope={"type": "http"})
    _request._url = URL(url="/update/key")

    await update_key_fn(data=request, request=_request)
    result = await info_key_fn(key=generated_key)
    print("result from info_key_fn", result)
    assert result["key"] == generated_key
    print("\n info for key=", result["info"])
    assert result["info"]["metadata"] == {
        "team": "litellm-team3",
        "model_tpm_limit": {"gpt-4": 200},
        "model_rpm_limit": {"gpt-4": 3},
    }


@pytest.mark.asyncio()
async def test_generate_key_with_guardrails(prisma_client):
    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    request = GenerateKeyRequest(
        guardrails=["aporia-pre-call"],
        metadata={
            "team": "litellm-team3",
        },
    )
    key = await generate_key_fn(
        data=request,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print("generated key=", key)

    generated_key = key.key

    # use generated key to auth in
    result = await info_key_fn(key=generated_key)
    print("result from info_key_fn", result)
    assert result["key"] == generated_key
    print("\n info for key=", result["info"])
    assert result["info"]["metadata"] == {
        "team": "litellm-team3",
        "guardrails": ["aporia-pre-call"],
    }

    # Update model tpm_limit and rpm_limit
    request = UpdateKeyRequest(
        key=generated_key,
        guardrails=["aporia-pre-call", "aporia-post-call"],
    )
    _request = Request(scope={"type": "http"})
    _request._url = URL(url="/update/key")

    await update_key_fn(data=request, request=_request)
    result = await info_key_fn(key=generated_key)
    print("result from info_key_fn", result)
    assert result["key"] == generated_key
    print("\n info for key=", result["info"])
    assert result["info"]["metadata"] == {
        "team": "litellm-team3",
        "guardrails": ["aporia-pre-call", "aporia-post-call"],
    }


@pytest.mark.asyncio()
async def test_team_access_groups(prisma_client):
    """
    Test team based model access groups

    - Test calling a model in the access group  -> pass
    - Test calling a model not in the access group -> fail
    """
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    # create router with access groups
    litellm_router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-pro-vision",
                "litellm_params": {
                    "model": "vertex_ai/gemini-1.0-pro-vision-001",
                },
                "model_info": {"access_groups": ["beta-models"]},
            },
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "gpt-4o",
                },
                "model_info": {"access_groups": ["beta-models"]},
            },
        ]
    )
    setattr(litellm.proxy.proxy_server, "llm_router", litellm_router)

    # Create team with models=["beta-models"]
    team_request = NewTeamRequest(
        team_alias="testing-team",
        models=["beta-models"],
    )

    new_team_response = await new_team(
        data=team_request,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        http_request=Request(scope={"type": "http"}),
    )
    print("new_team_response", new_team_response)
    created_team_id = new_team_response["team_id"]

    # create key with team_id=created_team_id
    request = GenerateKeyRequest(
        team_id=created_team_id,
    )

    key = await generate_key_fn(
        data=request,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print(key)

    generated_key = key.key
    bearer_token = "Bearer " + generated_key

    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    for model in ["gpt-4o", "gemini-pro-vision"]:
        # Expect these to pass
        async def return_body():
            return_string = f'{{"model": "{model}"}}'
            # return string as bytes
            return return_string.encode()

        request.body = return_body

        # use generated key to auth in
        print(
            "Bearer token being sent to user_api_key_auth() - {}".format(bearer_token)
        )
        result = await user_api_key_auth(request=request, api_key=bearer_token)

    for model in ["gpt-4", "gpt-4o-mini", "gemini-experimental"]:
        # Expect these to fail
        async def return_body_2():
            return_string = f'{{"model": "{model}"}}'
            # return string as bytes
            return return_string.encode()

        request.body = return_body_2

        # use generated key to auth in
        print(
            "Bearer token being sent to user_api_key_auth() - {}".format(bearer_token)
        )
        try:
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            pytest.fail(f"This should have failed!. IT's an invalid model")
        except Exception as e:
            print("got exception", e)
            assert (
                "not allowed to call model" in e.message
                and "Allowed team models" in e.message
            )


################ Unit Tests for testing regeneration of keys ###########
@pytest.mark.asyncio()
async def test_regenerate_api_key(prisma_client):
    litellm.set_verbose = True
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()
    import uuid

    # generate new key
    key_alias = f"test_alias_regenerate_key-{uuid.uuid4()}"
    spend = 100
    max_budget = 400
    models = ["fake-openai-endpoint"]
    new_key = await generate_key_fn(
        data=GenerateKeyRequest(
            key_alias=key_alias, spend=spend, max_budget=max_budget, models=models
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    generated_key = new_key.key
    print(generated_key)

    # assert the new key works as expected
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body():
        return_string = f'{{"model": "fake-openai-endpoint"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body
    result = await user_api_key_auth(request=request, api_key=f"Bearer {generated_key}")
    print(result)

    # regenerate the key
    new_key = await regenerate_key_fn(
        key=generated_key,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
    )
    print("response from regenerate_key_fn", new_key)

    # assert the new key works as expected
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body_2():
        return_string = f'{{"model": "fake-openai-endpoint"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body_2
    result = await user_api_key_auth(request=request, api_key=f"Bearer {new_key.key}")
    print(result)

    # assert the old key stops working
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    async def return_body_3():
        return_string = f'{{"model": "fake-openai-endpoint"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body_3
    try:
        result = await user_api_key_auth(
            request=request, api_key=f"Bearer {generated_key}"
        )
        print(result)
        pytest.fail(f"This should have failed!. the key has been regenerated")
    except Exception as e:
        print("got expected exception", e)
        assert "Invalid proxy server token passed" in e.message

    # Check that the regenerated key has the same spend, max_budget, models and key_alias
    assert new_key.spend == spend, f"Expected spend {spend} but got {new_key.spend}"
    assert (
        new_key.max_budget == max_budget
    ), f"Expected max_budget {max_budget} but got {new_key.max_budget}"
    assert (
        new_key.key_alias == key_alias
    ), f"Expected key_alias {key_alias} but got {new_key.key_alias}"
    assert (
        new_key.models == models
    ), f"Expected models {models} but got {new_key.models}"

    assert new_key.key_name == f"sk-...{new_key.key[-4:]}"

    pass
