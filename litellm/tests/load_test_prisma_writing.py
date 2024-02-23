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
import time

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


request_data = {
    "model": "azure-gpt-3.5",
    "messages": [
        {"role": "user", "content": "this is my new test. respond in 50 lines"}
    ],
}


async def test_call_with_key_never_over_budget():

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    # Reset litellm.proxy.proxy_server.prisma_client to None
    litellm.proxy.proxy_server.custom_db_client = None

    litellm.proxy.proxy_server.user_custom_key_generate = None
    # Make a call with a key with budget=None, it should never fail
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:
        await litellm.proxy.proxy_server.prisma_client.connect()
        request = GenerateKeyRequest(max_budget=10)
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
        import time

        kwargs = {
            "model": "chatgpt-v-2",
            "messages": [
                {"role": "user", "content": "write a para about yc and litellm"}
            ],
            "optional_params": {"extra_body": {}},
            "litellm_params": {
                "acompletion": True,
                "api_key": "d6f82361954b450188295b448e2091ca",
                "force_timeout": 600,
                "logger_fn": None,
                "verbose": False,
                "custom_llm_provider": "azure",
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "litellm_call_id": "17ca368a-fb0d-4d02-87f5-be66327b59b3",
                "model_alias_map": {},
                "completion_call_id": None,
                "metadata": {
                    "user_api_key": hash_token(generated_key),
                    "user_api_key_user_id": user_id,
                    "user_api_key_team_id": None,
                    "user_api_key_metadata": {},
                    "headers": {
                        "host": "0.0.0.0:4000",
                        "user-agent": "curl/7.88.1",
                        "accept": "*/*",
                        "content-type": "application/json",
                        "content-length": "196",
                    },
                    "endpoint": "http://0.0.0.0:4000/chat/completions",
                    "model_group": "azure-gpt-3.5",
                    "deployment": "azure/chatgpt-v-2",
                    "model_info": {
                        "mode": "chat",
                        "max_tokens": 4096,
                        "base_model": "azure/gpt-4-1106-preview",
                        "access_groups": ["public"],
                        "id": "5f837118-8045-4038-bceb-96213da4d3a8",
                    },
                    "caching_groups": None,
                },
                "model_info": {
                    "mode": "chat",
                    "max_tokens": 4096,
                    "base_model": "azure/gpt-4-1106-preview",
                    "access_groups": ["public"],
                    "id": "5f837118-8045-4038-bceb-96213da4d3a8",
                },
                "proxy_server_request": {
                    "url": "http://0.0.0.0:4000/chat/completions",
                    "method": "POST",
                    "headers": {
                        "host": "0.0.0.0:4000",
                        "user-agent": "curl/7.88.1",
                        "accept": "*/*",
                        "content-type": "application/json",
                        "authorization": "Bearer sk-1234",
                        "content-length": "196",
                    },
                    "body": {
                        "model": "azure-gpt-3.5",
                        "messages": [
                            {
                                "role": "user",
                                "content": "write a para about yc and litellm",
                            }
                        ],
                    },
                },
                "preset_cache_key": None,
                "stream_response": {},
            },
            "start_time": datetime(2024, 2, 22, 12, 53, 2, 565482),
            "stream": False,
            "user": None,
            "call_type": "acompletion",
            "litellm_call_id": "17ca368a-fb0d-4d02-87f5-be66327b59b3",
            "max_retries": 0,
            "extra_body": {},
            "input": [{"role": "user", "content": "write a para about yc and litellm"}],
            "api_key": "d6f82361954b450188295b448e2091ca",
            "additional_args": {
                "headers": {"Authorization": "Bearer d6f82361954b450188295b448e2091ca"},
                "api_base": True,
                "complete_input_dict": {
                    "model": "chatgpt-v-2",
                    "messages": [
                        {"role": "user", "content": "write a para about yc and litellm"}
                    ],
                    "extra_body": {},
                },
            },
            "log_event_type": "post_api_call",
            "end_time": datetime(2024, 2, 22, 12, 53, 6, 271876),
            "cache_hit": None,
            "response_cost": 0.00574,
        }

        # make 1k concurrent track_cost_callback requests
        for _ in range(10):
            num_requests = 100
            # Create a list to store the tasks
            tasks = []
            # Make 1k concurrent track_cost_callback requests
            for _ in range(num_requests):
                # add 5 random shars to resp.id
                import random

                request_id = f"chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac{str(random.randint(0, 1000))}"
                resp = ModelResponse(
                    id=request_id,
                    choices=[
                        Choices(
                            finish_reason=None,
                            index=0,
                            message=Message(
                                content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a"
                                * 500,
                                role="assistant",
                            ),
                        )
                    ],
                    model="gpt-35-turbo",  # azure always has model written like this
                    usage=Usage(
                        prompt_tokens=10, completion_tokens=21, total_tokens=31
                    ),
                )
                task = asyncio.create_task(
                    track_cost_callback(
                        kwargs=kwargs,
                        completion_response=resp,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                    )
                )
                tasks.append(task)

            # Gather and await all the tasks
            print("starting 1k concurrent track_cost_callback requests")
        await asyncio.gather(*tasks)
        print("done making 1k concurrent track_cost_callback requests")

        time.sleep(10)

        await asyncio.sleep(10)
        # use generated key to auth in
        result = await user_api_key_auth(request=request, api_key=bearer_token)
        print("result from user auth with new key", result)
    except Exception as e:
        print("Got an Exception", e)
        pass


# call test_call_with_key_never_over_budget

import asyncio

asyncio.run(test_call_with_key_never_over_budget())
