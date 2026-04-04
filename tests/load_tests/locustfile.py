"""
Locust load test for LiteLLM proxy.

Usage:
    # Start mock server:
    poetry run python tests/load_tests/mock_openai_server.py &

    # Start proxy:
    poetry run litellm --config tests/load_tests/loadtest_config.yaml --port 4000 &

    # Run headless load test (baseline):
    poetry run locust -f tests/load_tests/locustfile.py \
        --headless -u 200 -r 50 --run-time 30s \
        --host http://localhost:4000 \
        --csv results/baseline \
        --only-summary

    # Run with web UI:
    poetry run locust -f tests/load_tests/locustfile.py --host http://localhost:4000
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
