"""Locust user for the throughput load test, run as a subprocess by locust_load.py.

Reads the target model and bearer key from the environment (locust_load passes
LOAD_MODEL and LOAD_API_KEY; --host carries the proxy base URL) and hammers
/chat/completions with FastHttpUser, the high-throughput client Locust ships. The
model is a mock deployment, so every request exercises the full proxy path without
an upstream provider call.
"""

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
