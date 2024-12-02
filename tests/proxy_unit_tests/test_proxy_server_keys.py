# import sys, os, time
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os, io

# # this file is to test litellm/proxy

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest, logging
# import litellm
# from litellm import embedding, completion, completion_cost, Timeout
# from litellm import RateLimitError


# import sys, os, time
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os, io

# # this file is to test litellm/proxy
# from concurrent.futures import ThreadPoolExecutor

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path

# import pytest, logging, requests
# import litellm
# from litellm import embedding, completion, completion_cost, Timeout
# from litellm import RateLimitError
# from github import Github
# import subprocess


# # Function to execute a command and return the output
# def run_command(command):
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
#     output, _ = process.communicate()
#     return output.decode().strip()


# # Retrieve the current branch name
# branch_name = run_command("git rev-parse --abbrev-ref HEAD")

# # GitHub personal access token (with repo scope) or use username and password
# access_token = os.getenv("GITHUB_ACCESS_TOKEN")
# # Instantiate the PyGithub library's Github object
# g = Github(access_token)

# # Provide the owner and name of the repository where the pull request is located
# repository_owner = "BerriAI"
# repository_name = "litellm"

# # Get the repository object
# repo = g.get_repo(f"{repository_owner}/{repository_name}")

# # Iterate through the pull requests to find the one related to your branch
# for pr in repo.get_pulls():
#     print(f"in here! {pr.head.ref}")
#     if pr.head.ref == branch_name:
#         pr_number = pr.number
#         break

# print(f"The pull request number for branch {branch_name} is: {pr_number}")


# def test_add_new_key():
#     max_retries = 3
#     retry_delay = 10  # seconds

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

#             endpoint = f"https://litellm-litellm-pr-{pr_number}.up.railway.app"

#             # Make a request to the staging endpoint
#             response = requests.post(
#                 endpoint + "/key/generate", json=test_data, headers=headers
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


# def test_update_new_key():
#     try:
#         # Your test data
#         test_data = {
#             "models": ["gpt-3.5-turbo", "gpt-4", "claude-2", "azure-model"],
#             "aliases": {"mistral-7b": "gpt-3.5-turbo"},
#             "duration": "20m",
#         }
#         print("testing proxy server")
#         # Your bearer token
#         token = os.getenv("PROXY_MASTER_KEY")
#         headers = {"Authorization": f"Bearer {token}"}

#         endpoint = f"https://litellm-litellm-pr-{pr_number}.up.railway.app"

#         # Make a request to the staging endpoint
#         response = requests.post(
#             endpoint + "/key/generate", json=test_data, headers=headers
#         )
#         assert response.status_code == 200
#         result = response.json()
#         assert result["key"].startswith("sk-")

#         def _post_data():
#             json_data = {"models": ["bedrock-models"], "key": result["key"]}
#             response = requests.post(
#                 endpoint + "/key/generate", json=json_data, headers=headers
#             )
#             print(f"response text: {response.text}")
#             assert response.status_code == 200
#             return response

#         _post_data()
#         print(f"Received response: {result}")
#     except Exception as e:
#         pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")

# def test_add_new_key_max_parallel_limit():
#     try:
#         # Your test data
#         test_data = {"duration": "20m", "max_parallel_requests": 1}
#         # Your bearer token
#         token = os.getenv("PROXY_MASTER_KEY")
#         headers = {"Authorization": f"Bearer {token}"}

#         endpoint = f"https://litellm-litellm-pr-{pr_number}.up.railway.app"
#         print(f"endpoint: {endpoint}")
#         # Make a request to the staging endpoint
#         response = requests.post(
#             endpoint + "/key/generate", json=test_data, headers=headers
#         )
#         assert response.status_code == 200
#         result = response.json()

#         # load endpoint with model
#         model_data = {
#             "model_name": "azure-model",
#             "litellm_params": {
#                 "model": "azure/chatgpt-v-2",
#                 "api_key": os.getenv("AZURE_API_KEY"),
#                 "api_base": os.getenv("AZURE_API_BASE"),
#                 "api_version": os.getenv("AZURE_API_VERSION")
#             }
#         }
#         response = requests.post(endpoint + "/model/new", json=model_data, headers=headers)
#         assert response.status_code == 200
#         print(f"response text: {response.text}")


#         def _post_data():
#             json_data = {
#                 "model": "azure-model",
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": f"this is a test request, write a short poem {time.time()}",
#                     }
#                 ],
#             }
#             # Your bearer token
#             response = requests.post(
#                 endpoint + "/chat/completions", json=json_data, headers={"Authorization": f"Bearer {result['key']}"}
#             )
#             return response

#         def _run_in_parallel():
#             with ThreadPoolExecutor(max_workers=2) as executor:
#                 future1 = executor.submit(_post_data)
#                 future2 = executor.submit(_post_data)

#                 # Obtain the results from the futures
#                 response1 = future1.result()
#                 print(f"response1 text: {response1.text}")
#                 response2 = future2.result()
#                 print(f"response2 text: {response2.text}")
#                 if response1.status_code == 429 or response2.status_code == 429:
#                     pass
#                 else:
#                     raise Exception()

#         _run_in_parallel()
#     except Exception as e:
#         pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")

# def test_add_new_key_max_parallel_limit_streaming():
#     try:
#         # Your test data
#         test_data = {"duration": "20m", "max_parallel_requests": 1}
#         # Your bearer token
#         token = os.getenv("PROXY_MASTER_KEY")
#         headers = {"Authorization": f"Bearer {token}"}

#         endpoint = f"https://litellm-litellm-pr-{pr_number}.up.railway.app"

#         # Make a request to the staging endpoint
#         response = requests.post(
#             endpoint + "/key/generate", json=test_data, headers=headers
#         )
#         print(f"response: {response.text}")
#         assert response.status_code == 200
#         result = response.json()

#         def _post_data():
#             json_data = {
#                 "model": "azure-model",
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": f"this is a test request, write a short poem {time.time()}",
#                     }
#                 ],
#                 "stream": True,
#             }
#             response = requests.post(
#                 endpoint + "/chat/completions", json=json_data, headers={"Authorization": f"Bearer {result['key']}"}
#             )
#             return response

#         def _run_in_parallel():
#             with ThreadPoolExecutor(max_workers=2) as executor:
#                 future1 = executor.submit(_post_data)
#                 future2 = executor.submit(_post_data)

#                 # Obtain the results from the futures
#                 response1 = future1.result()
#                 response2 = future2.result()
#                 if response1.status_code == 429 or response2.status_code == 429:
#                     pass
#                 else:
#                     raise Exception()

#         _run_in_parallel()
#     except Exception as e:
#         pytest.fail(f"LiteLLM Proxy test failed. Exception: {str(e)}")
