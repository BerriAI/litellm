"""
Locust load test for performance comparison.
Uses a custom shape: 10s ramp to 5000 users, hold for 60s, then stop.
Results from the steady-state period are what matters.
"""
from locust import HttpUser, task, between


class ChatCompletionUser(HttpUser):
    wait_time = between(0.01, 0.02)

    @task
    def post_chat_completions(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        data = {
            "model": "fake-openai-endpoint",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.client.post("/chat/completions", json=data, headers=headers)
