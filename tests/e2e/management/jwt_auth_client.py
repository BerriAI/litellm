"""Client for the JWT-auth e2e suite: send a bearer JWT (issued by the real IdP,
or a deliberately untrusted one) at the gateway and read the raw outcome.

A JWT is just a bearer credential to the transport, so this is a thin wrapper that
routes a token at an arbitrary route (to prove admin access) or at chat (to prove
team-scoped model access). Team/model setup and verification go through the master
key via ManagementClient, so the suite injects both.
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_http import NoBody, Result, StreamingResponse
from models import ChatBody, ChatMessage
from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class JWTAuthClient:
    proxy: ProxyClient

    def get_route(self, path: str, token: str) -> Result[NoBody]:
        """GET `path` authenticated with `token`. A 200 -> Success, a 401 ->
        UnauthorizedError, a 403 -> UnknownApiError(403); the body is not modelled."""
        return self.proxy.transport.get(
            path,
            headers=self.proxy.transport.bearer(token),
            params=NoBody(),
            response_type=NoBody,
        )

    def chat(self, token: str, model: str, content: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(token),
            json=ChatBody(model=model, messages=[ChatMessage(role="user", content=content)], max_tokens=16),
        )


def build_jwt_client(proxy: ProxyClient) -> JWTAuthClient:
    return JWTAuthClient(proxy=proxy)
