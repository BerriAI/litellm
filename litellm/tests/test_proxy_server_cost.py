# #### What this tests ####
# #    This tests the cost tracking function works with consecutive calls (~10 consecutive calls)

# import sys, os, asyncio
# import traceback
# import pytest
# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import dotenv
# dotenv.load_dotenv()
# import litellm
# from fastapi.testclient import TestClient
# from fastapi import FastAPI
# from litellm.proxy.proxy_server import router, save_worker_config, startup_event  # Replace with the actual module where your FastAPI router is defined
# filepath = os.path.dirname(os.path.abspath(__file__))
# config_fp = f"{filepath}/test_config.yaml"
# save_worker_config(config=config_fp, model=None, alias=None, api_base=None, api_version=None, debug=True, temperature=None, max_tokens=None, request_timeout=600, max_budget=None, telemetry=False, drop_params=True, add_function_to_prompt=False, headers=None, save=False, use_queue=False)
# app = FastAPI()
# app.include_router(router)  # Include your router in the test app
# @app.on_event("startup")
# async def wrapper_startup_event():
#     await startup_event()

# # Here you create a fixture that will be used by your tests
# # Make sure the fixture returns TestClient(app)
# @pytest.fixture(autouse=True)
# def client():
#     with TestClient(app) as client:
#         yield client

# @pytest.mark.asyncio
# async def test_proxy_cost_tracking(client):
#     """
#     Get min cost.
#     Create new key.
#     Run 10 parallel calls.
#     Check cost for key at the end.
#     assert it's > min cost.
#     """
#     model = "gpt-3.5-turbo"
#     messages = [{"role": "user", "content": "Hey, how's it going?"}]
#     number_of_calls = 1
#     min_cost = litellm.completion_cost(model=model, messages=messages) * number_of_calls
#     try:
#         ### CREATE NEW KEY ###
#         test_data = {
#             "models": ["azure-model"],
#         }
#         # Your bearer token
#         token = os.getenv("PROXY_MASTER_KEY")

#         headers = {
#             "Authorization": f"Bearer {token}"
#         }
#         create_new_key = client.post("/key/generate", json=test_data, headers=headers)
#         key = create_new_key.json()["key"]
#         print(f"received key: {key}")
#         ### MAKE PARALLEL CALLS ###
#         async def test_chat_completions():
#             # Your test data
#             test_data = {
#                 "model": "azure-model",
#                 "messages": messages
#             }

#             tmp_headers = {
#                 "Authorization": f"Bearer {key}"
#             }

#             response = client.post("/v1/chat/completions", json=test_data, headers=tmp_headers)

#             assert response.status_code == 200
#             result = response.json()
#             print(f"Received response: {result}")
#         tasks = [test_chat_completions() for _ in range(number_of_calls)]
#         chat_completions = await asyncio.gather(*tasks)
#         ### CHECK SPEND ###
#         get_key_spend = client.get(f"/key/info?key={key}", headers=headers)

#         assert get_key_spend.json()["info"]["spend"] > min_cost
# #         print(f"chat_completions: {chat_completions}")
# #     except Exception as e:
# #         pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")

# #### JUST TEST LOCAL PROXY SERVER

# import requests, os
# from concurrent.futures import ThreadPoolExecutor
# import dotenv
# dotenv.load_dotenv()

# api_url = "http://0.0.0.0:8000/chat/completions"

# def make_api_call(api_url):
#     # Your test data
#     test_data = {
#         "model": "azure-model",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "hi"
#             },
#         ],
#         "max_tokens": 10,
#     }
#     # Your bearer token
#     token = os.getenv("PROXY_MASTER_KEY")

#     headers = {
#         "Authorization": f"Bearer {token}"
#     }
#     print("testing proxy server")
#     response = requests.post(api_url, json=test_data, headers=headers)
#     return response.json()

# # Number of parallel API calls
# num_parallel_calls = 3

# # List to store results
# results = []

# # Create a ThreadPoolExecutor
# with ThreadPoolExecutor() as executor:
#     # Submit the API calls concurrently
#     futures = [executor.submit(make_api_call, api_url) for _ in range(num_parallel_calls)]

#     # Gather the results as they become available
#     for future in futures:
#         try:
#             result = future.result()
#             results.append(result)
#         except Exception as e:
#             print(f"Error: {e}")

# # Print the results
# for idx, result in enumerate(results, start=1):
#     print(f"Result {idx}: {result}")
