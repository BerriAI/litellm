# Test the following scenarios:
# 1. Generate a Key, and use it to make a call
# 2. Make a call with invalid key, expect it to fail
# 3. Make a call to a key with invalid model - expect to fail
# 4. Make a call to a key with valid model - expect to pass
# 5. Make a call with expired key - expect to fail
# 6. Make a call with unexpired key - expect to pass
# 7. Make a call with key under budget, expect to pass
# 8. Make a call with key over budget, expect to fail


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
from litellm.proxy.proxy_server import new_user, user_api_key_auth

from litellm.proxy._types import NewUserRequest, DynamoDBArgs
from litellm.proxy.utils import DBClient
from starlette.datastructures import URL

db_args = {
    "ssl_verify": False,
    "billing_mode": "PAY_PER_REQUEST",
    "region_name": "us-west-2",
}
custom_db_client = DBClient(
    custom_db_type="dynamo_db",
    custom_db_args=db_args,
)

request_data = {
    "model": "azure-gpt-3.5",
    "messages": [
        {"role": "user", "content": "this is my new test. respond in 50 lines"}
    ],
}


def test_generate_and_call_with_valid_key():
    # 1. Generate a Key, and use it to make a call
    setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
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


def test_call_with_invalid_key():
    # 2. Make a call with invalid key, expect it to fail
    setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    try:

        async def test():
            generated_key = "bad-key"
            bearer_token = "Bearer " + generated_key

            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            # use generated key to auth in
            result = await user_api_key_auth(request=request, api_key=bearer_token)
            pytest.fail(f"This should have failed!")

        asyncio.run(test())
    except Exception as e:
        print("Got Exception", e)
        print(str(e))
        pass


# def test_call_with_invalid_model():
#     # 3. Make a call to a key with an invalid model - expect to fail
#     key = new_user(ValidNewUserRequest())
#     result = user_auth(InvalidModelRequest(key))
#     assert result is False


# def test_call_with_valid_model():
#     # 4. Make a call to a key with a valid model - expect to pass
#     key = new_user(ValidNewUserRequest())
#     result = user_auth(ValidModelRequest(key))
#     assert result is True


# def test_call_with_expired_key():
#     # 5. Make a call with an expired key - expect to fail
#     key = new_user(ExpiredKeyRequest())
#     result = user_auth(ValidRequest(key))
#     assert result is False


# def test_call_with_unexpired_key():
#     # 6. Make a call with an unexpired key - expect to pass
#     key = new_user(UnexpiredKeyRequest())
#     result = user_auth(ValidRequest(key))
#     assert result is True


# def test_call_with_key_under_budget():
#     # 7. Make a call with a key under budget, expect to pass
#     key = new_user(KeyUnderBudgetRequest())
#     result = user_auth(ValidRequest(key))
#     assert result is True


# def test_call_with_key_over_budget():
#     # 8. Make a call with a key over budget, expect to fail
#     key = new_user(KeyOverBudgetRequest())
#     result = user_auth(ValidRequest(key))
#     assert result is False
