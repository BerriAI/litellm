"""Client for the quota-management suite: the shared Gateway plus raw chat
calls judged by HTTP status, body, and headers (a rate-limit block is a 429
whose body and retry-after header carry the contract, not a typed success
model)."""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway
from e2e_http import StreamingResponse
from models import ChatBody, ChatMessage


@dataclass(frozen=True, slots=True)
class QuotaClient:
    gateway: Gateway

    def chat(self, key: str, model: str, content: str, *, max_tokens: int = 16) -> StreamingResponse:
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
            ),
        )


def build_client() -> QuotaClient:
    return QuotaClient(gateway=build_gateway())
