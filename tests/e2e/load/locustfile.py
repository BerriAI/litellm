from __future__ import annotations

import os
import random
import uuid
from typing import Final

from locust import FastHttpUser, constant, task

_MODEL: Final = os.environ["LOAD_MODEL"]
_API_KEYS: Final = tuple(os.environ["LOAD_API_KEYS"].split(","))


def _payload() -> dict[str, object]:
    """A prompt no other request sent, so the response cache never answers for the deployment."""
    return {
        "model": _MODEL,
        "messages": [{"role": "user", "content": f"load test ping {uuid.uuid4().hex}"}],
        "max_tokens": 16,
    }


class ChatUser(FastHttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        self.headers = {"Authorization": f"Bearer {random.choice(_API_KEYS)}"}

    @task
    def chat(self) -> None:
        self.client.post(  # pyright: ignore[reportUnknownMemberType]  # locust FastHttpSession.post types json/**kwargs as Any
            "/chat/completions",
            json=_payload(),
            headers=self.headers,
            name="/chat/completions",
        )
