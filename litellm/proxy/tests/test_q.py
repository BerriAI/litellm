import requests
import time

# Step 1 Add a config to the proxy, generate a temp key 
config = {

}

response = requests.post(
    url = "http://0.0.0.0:8000/key/generate",
    json={
        "config": config,
        "duration": "30d" # default to 30d, set it to 30m if you want a temp key
    },
    headers={
        "Authorization": "Bearer sk-hosted-litellm"
    }
)

print("\nresponse from generating key", response.json())

generated_key = response.json()["key"]
print("\ngenerated key for proxy", generated_key)


# Step 2: Queue a request to the proxy, using your generated_key
job_response = requests.post(
    url = "http://0.0.0.0:8000/queue/request",
    json={
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'system', 'content': f'You are a helpful assistant. What is your name'},
            ],
    },
    headers={
        "Authorization": f"Bearer {generated_key}"
    }
)

job_response = job_response.json()
job_id  = job_response["id"]
polling_url = job_response["url"]
polling_url = f"http://0.0.0.0:8000{polling_url}"
print("\nCreated Job, Polling Url", polling_url)

# Step 3: Poll the request
while True:
    try:
        print("\nPolling URL", polling_url)
        polling_response = requests.get(
                url=polling_url, 
                headers={
                    "Authorization": f"Bearer {generated_key}"
                }
            )
        polling_response = polling_response.json()
        print("\nResponse from polling url", polling_response)
        status = polling_response["status"]
        if status == "finished":
            llm_response = polling_response["result"]
            print("LLM Response")
            print(llm_response)
            break
        time.sleep(0.5)
    except Exception as e:
        print("got exception in polling", e)
        break





        
