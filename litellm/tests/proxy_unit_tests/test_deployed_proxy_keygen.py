# import sys, os, time
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os, io

# # this file is to test litellm/proxy

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest, logging, requests
# import litellm
# from litellm import embedding, completion, completion_cost, Timeout
# from litellm import RateLimitError


# def test_add_new_key():
#     max_retries = 3
#     retry_delay = 1  # seconds

#     for retry in range(max_retries + 1):
#         try:
#             # Your test data
#             test_data = {
#                 "models": ["gpt-3.5-turbo", "gpt-4", "claude-2", "azure-model"],
#                 "aliases": {"mistral-7b": "gpt-3.5-turbo"},
#                 "duration": "20m",
#             }
#             print("testing proxy server")

#             # Your bearer token
#             token = os.getenv("PROXY_MASTER_KEY")
#             headers = {"Authorization": f"Bearer {token}"}

#             staging_endpoint = "https://litellm-litellm-pr-1376.up.railway.app"
#             main_endpoint = "https://litellm-staging.up.railway.app"

#             # Make a request to the staging endpoint
#             response = requests.post(
#                 main_endpoint + "/key/generate", json=test_data, headers=headers
#             )

#             print(f"response: {response.text}")

#             if response.status_code == 200:
#                 result = response.json()
#                 break  # Successful response, exit the loop
#             elif response.status_code == 503 and retry < max_retries:
#                 print(
#                     f"Retrying in {retry_delay} seconds... (Retry {retry + 1}/{max_retries})"
#                 )
#                 time.sleep(retry_delay)
#             else:
#                 assert False, f"Unexpected response status code: {response.status_code}"

#         except Exception as e:
#             print(traceback.format_exc())
#             pytest.fail(f"An error occurred {e}")


# test_add_new_key()
