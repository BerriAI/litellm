import json
import time
import uuid

from locust import HttpUser, between, events, task


class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def chat_completion(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
            # Include any additional headers you may need for authentication, etc.
        }

        # Customize the payload with "model" and "messages" keys
        payload = {
            "model": "fake-openai-endpoint",
            "messages": [
                {
                    "role": "system",
                    "content": f"{uuid.uuid4()} this is a very sweet test message from ishaan"
                    * 100,
                },
                {"role": "user", "content": "Hello, how are you?"},
            ],
            # Add more data as necessary
        }

        # Make a POST request to the "chat/completions" endpoint
        self.client.post("chat/completions", json=payload, headers=headers)

        # Print or log the response if needed
