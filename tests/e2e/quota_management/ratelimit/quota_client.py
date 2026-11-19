"""Client for the quota-management suite: the shared ProxyClient plus raw chat
calls judged by HTTP status, body, and headers (a rate-limit block is a 429
whose body and retry-after header carry the contract, not a typed success
model)."""

from __future__ import annotations

from dataclasses import dataclass

from proxy_client import ProxyClient
from e2e_http import StreamingResponse, unwrap
from e2e_metadata import step
from models import ChatBody, ChatMessage, TeamData, TeamInfoParams, TeamInfoResponse


@dataclass(frozen=True, slots=True)
class QuotaClient:
    proxy: ProxyClient

    @step('Send a /chat/completions request to {model} with the prompt "{content}"')
    def chat(self, key: str, model: str, content: str, *, max_tokens: int = 16) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
            ),
        )

    @step('Send a streaming /chat/completions request to {model} with the prompt "{content}"')
    def stream_chat(self, key: str, model: str, content: str, *, max_tokens: int) -> StreamingResponse:
        return self.proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
                stream=True,
            ),
        )

    @step("Read the team back from /team/info")
    def team_info(self, team_id: str) -> TeamData:
        return unwrap(
            self.proxy.transport.get(
                "/team/info",
                headers=self.proxy.management_headers(),
                params=TeamInfoParams(team_id=team_id),
                response_type=TeamInfoResponse,
            )
        ).team_info


def build_client(proxy: ProxyClient) -> QuotaClient:
    return QuotaClient(proxy=proxy)
