from __future__ import annotations

import os
import random
import uuid
from itertools import cycle
from typing import Final

from locust import FastHttpUser, constant, task

_MODEL: Final = os.environ["LOAD_MODEL"]
_API_KEYS: Final = tuple(os.environ["LOAD_API_KEYS"].split(","))
_NEXT_ENDPOINT: Final = cycle(os.environ["LOAD_ENDPOINTS"].split(","))
_FILLER: Final = "x" * 40_000


def _payload(endpoint: str) -> dict[str, object]:
    """A prompt no other request sent, so the response cache never answers for the deployment.

    /chat/completions and /v1/messages take the same body: /v1/messages requires max_tokens,
    which /chat/completions also accepts. /v1/responses takes the prompt as `input` and the
    limit as `max_output_tokens`. Padded to tens of KB so a per-request bookkeeping cost that
    scales with body size (string formatting, hashing) shows up in the CPU and log-size budgets
    instead of hiding behind a 40-byte prompt.
    """
    prompt: Final = f"load test ping {uuid.uuid4().hex} {_FILLER}"
    if endpoint.endswith("/responses"):
        return {"model": _MODEL, "input": prompt, "max_output_tokens": 16}
    return {"model": _MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16}


class GatewayUser(FastHttpUser):
    """One simulated user, pinned to one endpoint for its lifetime.

    Endpoints are handed out round robin as users spawn, so a run spreads evenly over them
    while each user's traffic stays on a single route, the way a real client behaves.
    """

    wait_time = constant(0)

    def on_start(self) -> None:
        self.headers = {"Authorization": f"Bearer {random.choice(_API_KEYS)}"}
        self.endpoint = next(_NEXT_ENDPOINT)

    @task
    def call(self) -> None:
        self.client.post(  # pyright: ignore[reportUnknownMemberType]  # locust FastHttpSession.post types json/**kwargs as Any
            self.endpoint,
            json=_payload(self.endpoint),
            headers=self.headers,
            name=self.endpoint,
        )
