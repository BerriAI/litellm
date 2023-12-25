# #### What this tests ####
# #    This tests the router's ability to identify the least busy deployment

# #
# # How is this achieved?
# # - Before each call, have the router print the state of requests {"deployment": "requests_in_flight"}
# # - use litellm.input_callbacks to log when a request is just about to be made to a model - {"deployment-id": traffic}
# # - use litellm.success + failure callbacks to log when a request completed
# # - in get_available_deployment, for a given model group name -> pick based on traffic

# import sys, os, asyncio, time
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest
# from litellm import Router
# import litellm

# async def test_least_busy_routing():
#     model_list = [{
#         "model_name": "azure-model",
#         "litellm_params": {
#             "model": "azure/gpt-turbo",
#             "api_key": "os.environ/AZURE_FRANCE_API_KEY",
#             "api_base": "https://openai-france-1234.openai.azure.com",
#             "rpm": 1440,
#         }
#     }, {
#         "model_name": "azure-model",
#         "litellm_params": {
#             "model": "azure/gpt-35-turbo",
#             "api_key": "os.environ/AZURE_EUROPE_API_KEY",
#             "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
#             "rpm": 6
#         }
#     }, {
#         "model_name": "azure-model",
#         "litellm_params": {
#             "model": "azure/gpt-35-turbo",
#             "api_key": "os.environ/AZURE_CANADA_API_KEY",
#             "api_base": "https://my-endpoint-canada-berri992.openai.azure.com",
#             "rpm": 6
#         }
#     }]
#     router = Router(model_list=model_list,
# 					routing_strategy="least-busy",
# 					set_verbose=False,
#                     num_retries=3) # type: ignore

#     async def call_azure_completion():
#         try:
#             response = await router.acompletion(
#                 model="azure-model",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": "hello this request will pass"
#                     }
#                 ]
#             )
#             print("\n response", response)
#             return response
#         except:
#             return None

#     n = 1000
#     start_time = time.time()
#     tasks = [call_azure_completion() for _ in range(n)]
#     chat_completions = await asyncio.gather(*tasks)
#     successful_completions = [c for c in chat_completions if c is not None]
#     print(n, time.time() - start_time, len(successful_completions))

# asyncio.run(test_least_busy_routing())
