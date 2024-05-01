import requests
from concurrent.futures import ThreadPoolExecutor

# Replace the URL with your actual endpoint
url = "http://localhost:8000/router_acompletion"


def make_request(session):
    headers = {"Content-Type": "application/json"}
    data = {}  # Replace with your JSON payload if needed

    response = session.post(url, headers=headers, json=data)
    print(f"Status code: {response.status_code}")


# Number of concurrent requests
num_requests = 20

# Create a session to reuse the underlying TCP connection
with requests.Session() as session:
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        # Use list comprehension to submit tasks
        futures = [executor.submit(make_request, session) for _ in range(num_requests)]

        # Wait for all futures to complete
        for future in futures:
            future.result()
