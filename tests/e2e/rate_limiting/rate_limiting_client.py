"""Client for rate-limiting e2e checks."""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway
from e2e_http import StreamingResponse
from models import ChatBody, ChatMessage


@dataclass(frozen=True, slots=True)
class RateLimitingClient:
    gateway: Gateway

    def chat_status(
        self, key: str, model: str, content: str, max_tokens: int = 8
    ) -> StreamingResponse:
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
            ),
        )


def build_client() -> RateLimitingClient:
    return RateLimitingClient(gateway=build_gateway())
