# @pytest.mark.skip(reason="AWS Suspended Account")
# # Test the following scenarios:
# # 1. Generate a Key, and use it to make a call
# # 2. Make a call with invalid key, expect it to fail
# # 3. Make a call to a key with invalid model - expect to fail
# # 4. Make a call to a key with valid model - expect to pass
# # 5. Make a call with key over budget, expect to fail
# # 6. Make a streaming chat/completions call with key over budget, expect to fail


# # function to call to generate key - async def new_user(data: NewUserRequest):
# # function to validate a request - async def user_auth(request: Request):

# import sys, os
# import traceback
# from dotenv import load_dotenv
# from fastapi import Request

# load_dotenv()
# import os, io

# # this file is to test litellm/proxy

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest, logging, asyncio
# import litellm, asyncio
# from litellm.proxy.proxy_server import (
#     new_user,
#     user_api_key_auth,
#     user_update,
#     generate_key_fn,
# )

# from litellm.proxy._types import NewUserRequest, DynamoDBArgs, GenerateKeyRequest
# from litellm.proxy.utils import DBClient, hash_token
# from starlette.datastructures import URL


# request_data = {
#     "model": "azure-gpt-3.5",
#     "messages": [
#         {"role": "user", "content": "this is my new test. respond in 50 lines"}
#     ],
# }


# @pytest.fixture
# def custom_db_client():
#     # Assuming DBClient is a class that needs to be instantiated
#     db_args = {
#         "ssl_verify": False,
#         "billing_mode": "PAY_PER_REQUEST",
#         "region_name": "us-west-2",
#     }
#     custom_db_client = DBClient(
#         custom_db_type="dynamo_db",
#         custom_db_args=db_args,
#     )
#     # Reset litellm.proxy.proxy_server.prisma_client to None
#     litellm.proxy.proxy_server.prisma_client = None

#     return custom_db_client


# def test_generate_and_call_with_valid_key(custom_db_client):
#     # 1. Generate a Key, and use it to make a call
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     try:

#         async def test():
#             request = NewUserRequest()
#             key = await new_user(request)
#             print(key)

#             generated_key = key.key
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#         asyncio.run(test())
#     except Exception as e:
#         pytest.fail(f"An exception occurred - {str(e)}")


# def test_call_with_invalid_key(custom_db_client):
#     # 2. Make a call with invalid key, expect it to fail
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     try:

#         async def test():
#             generated_key = "bad-key"
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"}, receive=None)
#             request._url = URL(url="/chat/completions")

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             pytest.fail(f"This should have failed!. IT's an invalid key")

#         asyncio.run(test())
#     except Exception as e:
#         print("Got Exception", e)
#         print(e.message)
#         assert "Authentication Error" in e.message
#         pass


# def test_call_with_invalid_model(custom_db_client):
#     # 3. Make a call to a key with an invalid model - expect to fail
#     from litellm._logging import verbose_proxy_logger
#     import logging

#     verbose_proxy_logger.setLevel(logging.DEBUG)
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     try:

#         async def test():
#             request = NewUserRequest(models=["mistral"])
#             key = await new_user(request)
#             print(key)

#             generated_key = key.key
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             async def return_body():
#                 return b'{"model": "gemini-pro-vision"}'

#             request.body = return_body

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             pytest.fail(f"This should have failed!. IT's an invalid model")

#         asyncio.run(test())
#     except Exception as e:
#         assert (
#             e.message
#             == "Authentication Error, API Key not allowed to access model. This token can only access models=['mistral']. Tried to access gemini-pro-vision"
#         )
#         pass


# def test_call_with_valid_model(custom_db_client):
#     # 4. Make a call to a key with a valid model - expect to pass
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     try:

#         async def test():
#             request = NewUserRequest(models=["mistral"])
#             key = await new_user(request)
#             print(key)

#             generated_key = key.key
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             async def return_body():
#                 return b'{"model": "mistral"}'

#             request.body = return_body

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#         asyncio.run(test())
#     except Exception as e:
#         pytest.fail(f"An exception occurred - {str(e)}")


# def test_call_with_user_over_budget(custom_db_client):
#     # 5. Make a call with a key over budget, expect to fail
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     from litellm._logging import verbose_proxy_logger, verbose_logger
#     import logging

#     litellm.set_verbose = True
#     verbose_logger.setLevel(logging.DEBUG)
#     verbose_proxy_logger.setLevel(logging.DEBUG)
#     try:

#         async def test():
#             request = NewUserRequest(max_budget=0.00001)
#             key = await new_user(request)
#             print(key)

#             generated_key = key.key
#             user_id = key.user_id
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#             # update spend using track_cost callback, make 2nd request, it should fail
#             from litellm.proxy.proxy_server import (
#                 _PROXY_track_cost_callback as track_cost_callback,
#             )
#             from litellm import ModelResponse, Choices, Message, Usage

#             resp = ModelResponse(
#                 id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
#                 choices=[
#                     Choices(
#                         finish_reason=None,
#                         index=0,
#                         message=Message(
#                             content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
#                             role="assistant",
#                         ),
#                     )
#                 ],
#                 model="gpt-35-turbo",  # azure always has model written like this
#                 usage=Usage(prompt_tokens=210, completion_tokens=200, total_tokens=410),
#             )
#             await track_cost_callback(
#                 kwargs={
#                     "stream": False,
#                     "litellm_params": {
#                         "metadata": {
#                             "user_api_key": hash_token(generated_key),
#                             "user_api_key_user_id": user_id,
#                         }
#                     },
#                     "response_cost": 0.00002,
#                 },
#                 completion_response=resp,
#             )
#             await asyncio.sleep(5)
#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)
#             pytest.fail(f"This should have failed!. They key crossed it's budget")

#         asyncio.run(test())
#     except Exception as e:
#         error_detail = e.message
#         assert "Authentication Error, ExceededBudget:" in error_detail
#         print(vars(e))


# def test_call_with_user_over_budget_stream(custom_db_client):
#     # 6. Make a call with a key over budget, expect to fail
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     from litellm._logging import verbose_proxy_logger
#     import logging

#     litellm.set_verbose = True
#     verbose_proxy_logger.setLevel(logging.DEBUG)
#     try:

#         async def test():
#             request = NewUserRequest(max_budget=0.00001)
#             key = await new_user(request)
#             print(key)

#             generated_key = key.key
#             user_id = key.user_id
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#             # update spend using track_cost callback, make 2nd request, it should fail
#             from litellm.proxy.proxy_server import (
#                 _PROXY_track_cost_callback as track_cost_callback,
#             )
#             from litellm import ModelResponse, Choices, Message, Usage

#             resp = ModelResponse(
#                 id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
#                 choices=[
#                     Choices(
#                         finish_reason=None,
#                         index=0,
#                         message=Message(
#                             content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
#                             role="assistant",
#                         ),
#                     )
#                 ],
#                 model="gpt-35-turbo",  # azure always has model written like this
#                 usage=Usage(prompt_tokens=210, completion_tokens=200, total_tokens=410),
#             )
#             await track_cost_callback(
#                 kwargs={
#                     "stream": True,
#                     "complete_streaming_response": resp,
#                     "litellm_params": {
#                         "metadata": {
#                             "user_api_key": hash_token(generated_key),
#                             "user_api_key_user_id": user_id,
#                         }
#                     },
#                     "response_cost": 0.00002,
#                 },
#                 completion_response=ModelResponse(),
#             )
#             await asyncio.sleep(5)
#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)
#             pytest.fail(f"This should have failed!. They key crossed it's budget")

#         asyncio.run(test())
#     except Exception as e:
#         error_detail = e.message
#         assert "Authentication Error, ExceededBudget:" in error_detail
#         print(vars(e))


# def test_call_with_user_key_budget(custom_db_client):
#     # 7. Make a call with a key over budget, expect to fail
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     from litellm._logging import verbose_proxy_logger
#     import logging

#     verbose_proxy_logger.setLevel(logging.DEBUG)
#     try:

#         async def test():
#             request = GenerateKeyRequest(max_budget=0.00001)
#             key = await generate_key_fn(request)
#             print(key)

#             generated_key = key.key
#             user_id = key.user_id
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#             # update spend using track_cost callback, make 2nd request, it should fail
#             from litellm.proxy.proxy_server import (
#                 _PROXY_track_cost_callback as track_cost_callback,
#             )
#             from litellm import ModelResponse, Choices, Message, Usage

#             resp = ModelResponse(
#                 id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
#                 choices=[
#                     Choices(
#                         finish_reason=None,
#                         index=0,
#                         message=Message(
#                             content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
#                             role="assistant",
#                         ),
#                     )
#                 ],
#                 model="gpt-35-turbo",  # azure always has model written like this
#                 usage=Usage(prompt_tokens=210, completion_tokens=200, total_tokens=410),
#             )
#             await track_cost_callback(
#                 kwargs={
#                     "stream": False,
#                     "litellm_params": {
#                         "metadata": {
#                             "user_api_key": hash_token(generated_key),
#                             "user_api_key_user_id": user_id,
#                         }
#                     },
#                     "response_cost": 0.00002,
#                 },
#                 completion_response=resp,
#             )
#             await asyncio.sleep(5)
#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)
#             pytest.fail(f"This should have failed!. They key crossed it's budget")

#         asyncio.run(test())
#     except Exception as e:
#         error_detail = e.message
#         assert "Authentication Error, ExceededTokenBudget:" in error_detail
#         print(vars(e))


# def test_call_with_key_over_budget_stream(custom_db_client):
#     # 8. Make a call with a key over budget, expect to fail
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     from litellm._logging import verbose_proxy_logger
#     import logging

#     litellm.set_verbose = True
#     verbose_proxy_logger.setLevel(logging.DEBUG)
#     try:

#         async def test():
#             request = GenerateKeyRequest(max_budget=0.00001)
#             key = await generate_key_fn(request)
#             print(key)

#             generated_key = key.key
#             user_id = key.user_id
#             bearer_token = "Bearer " + generated_key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#             # update spend using track_cost callback, make 2nd request, it should fail
#             from litellm.proxy.proxy_server import (
#                 _PROXY_track_cost_callback as track_cost_callback,
#             )
#             from litellm import ModelResponse, Choices, Message, Usage

#             resp = ModelResponse(
#                 id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
#                 choices=[
#                     Choices(
#                         finish_reason=None,
#                         index=0,
#                         message=Message(
#                             content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
#                             role="assistant",
#                         ),
#                     )
#                 ],
#                 model="gpt-35-turbo",  # azure always has model written like this
#                 usage=Usage(prompt_tokens=210, completion_tokens=200, total_tokens=410),
#             )
#             await track_cost_callback(
#                 kwargs={
#                     "stream": True,
#                     "complete_streaming_response": resp,
#                     "litellm_params": {
#                         "metadata": {
#                             "user_api_key": hash_token(generated_key),
#                             "user_api_key_user_id": user_id,
#                         }
#                     },
#                     "response_cost": 0.00002,
#                 },
#                 completion_response=ModelResponse(),
#             )
#             await asyncio.sleep(5)
#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)
#             pytest.fail(f"This should have failed!. They key crossed it's budget")

#         asyncio.run(test())
#     except Exception as e:
#         error_detail = e.message
#         assert "Authentication Error, ExceededTokenBudget:" in error_detail
#         print(vars(e))


# def test_dynamo_db_migration(custom_db_client):
#     # Tests the temporary patch we have in place
#     setattr(litellm.proxy.proxy_server, "custom_db_client", custom_db_client)
#     setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
#     setattr(litellm.proxy.proxy_server, "user_custom_auth", None)
#     try:

#         async def test():
#             request = GenerateKeyRequest(max_budget=1)
#             key = await generate_key_fn(request)
#             print(key)

#             generated_key = key.key
#             bearer_token = (
#                 "Bearer " + generated_key
#             )  # this works with ishaan's db, it's a never expiring key

#             request = Request(scope={"type": "http"})
#             request._url = URL(url="/chat/completions")

#             async def return_body():
#                 return b'{"model": "azure-models"}'

#             request.body = return_body

#             # use generated key to auth in
#             result = await user_api_key_auth(request=request, api_key=bearer_token)
#             print("result from user auth with new key", result)

#         asyncio.run(test())
#     except Exception as e:
#         pytest.fail(f"An exception occurred - {traceback.format_exc()}")
