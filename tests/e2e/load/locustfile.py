from __future__ import annotations

import os

from locust import FastHttpUser, constant, task

_MODEL = os.environ["LOAD_MODEL"]
_HEADERS = {"Authorization": f"Bearer {os.environ['LOAD_API_KEY']}"}
_PAYLOAD = {
    "model": _MODEL,
    "messages": [{"role": "user", "content": "load test ping"}],
    "temperature": 0,
    "max_tokens": 16,
}


class ChatUser(FastHttpUser):
    wait_time = constant(0)

    @task
    def chat(self) -> None:
        self.client.post(  # pyright: ignore[reportUnknownMemberType]  # locust FastHttpSession.post types json/**kwargs as Any
            "/chat/completions",
            json=_PAYLOAD,
            headers=_HEADERS,
            name="/chat/completions",
        )
