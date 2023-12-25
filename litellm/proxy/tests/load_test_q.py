import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()


# Set the base URL as needed
base_url = "https://api.litellm.ai"
# # Uncomment the line below if you want to switch to the local server
# base_url = "http://0.0.0.0:8000"

# Step 1 Add a config to the proxy, generate a temp key
config = {
    "model_list": [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.environ["OPENAI_API_KEY"],
            },
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-2",
                "api_key": os.environ["AZURE_API_KEY"],
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "api_version": "2023-07-01-preview",
            },
        },
    ]
}
print("STARTING LOAD TEST Q")
print(os.environ["AZURE_API_KEY"])

response = requests.post(
    url=f"{base_url}/key/generate",
    json={
        "config": config,
        "duration": "30d",  # default to 30d, set it to 30m if you want a temp key
    },
    headers={"Authorization": "Bearer sk-hosted-litellm"},
)

print("\nresponse from generating key", response.text)
print("\n json response from gen key", response.json())

generated_key = response.json()["key"]
print("\ngenerated key for proxy", generated_key)


# Step 2: Queue 50 requests to the proxy, using your generated_key

import concurrent.futures


def create_job_and_poll(request_num):
    print(f"Creating a job on the proxy for request {request_num}")
    job_response = requests.post(
        url=f"{base_url}/queue/request",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "write a short poem"},
            ],
        },
        headers={"Authorization": f"Bearer {generated_key}"},
    )
    print(job_response.status_code)
    print(job_response.text)
    print("\nResponse from creating job", job_response.text)
    job_response = job_response.json()
    job_id = job_response["id"]
    polling_url = job_response["url"]
    polling_url = f"{base_url}{polling_url}"
    print(f"\nCreated Job {request_num}, Polling Url {polling_url}")

    # Poll each request
    while True:
        try:
            print(f"\nPolling URL for request {request_num}", polling_url)
            polling_response = requests.get(
                url=polling_url, headers={"Authorization": f"Bearer {generated_key}"}
            )
            print(
                f"\nResponse from polling url for request {request_num}",
                polling_response.text,
            )
            polling_response = polling_response.json()
            status = polling_response.get("status", None)
            if status == "finished":
                llm_response = polling_response["result"]
                print(f"LLM Response for request {request_num}")
                print(llm_response)
                # Write the llm_response to load_test_log.txt
                try:
                    with open("load_test_log.txt", "a") as response_file:
                        response_file.write(
                            f"Response for request: {request_num}\n{llm_response}\n\n"
                        )
                except Exception as e:
                    print("GOT EXCEPTION", e)
                break
            time.sleep(0.5)
        except Exception as e:
            print("got exception when polling", e)


# Number of requests
num_requests = 100

# Use ThreadPoolExecutor for parallel execution
with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
    # Create and poll each request in parallel
    futures = [executor.submit(create_job_and_poll, i) for i in range(num_requests)]

    # Wait for all futures to complete
    concurrent.futures.wait(futures)
