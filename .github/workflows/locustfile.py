from locust import HttpUser, task, between, events
import json
import time


class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def chat_completion(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer sk-S2-EZTUUDY0EmM6-Fy0Fyw",
            # Include any additional headers you may need for authentication, etc.
        }

        # Customize the payload with "model" and "messages" keys
        payload = {
            "model": "fake-openai-endpoint",
            "messages": [
                {"role": "system", "content": "You are a chat bot."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            # Add more data as necessary
        }

        # Make a POST request to the "chat/completions" endpoint
        response = self.client.post("chat/completions", json=payload, headers=headers)

        # Print or log the response if needed

    @task(10)
    def health_readiness(self):
        start_time = time.time()
        response = self.client.get("health/readiness")
        response_time = time.time() - start_time

    @task(10)
    def health_liveliness(self):
        start_time = time.time()
        response = self.client.get("health/liveliness")
        response_time = time.time() - start_time
