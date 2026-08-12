"""Client for the access-control e2e suite."""

from __future__ import annotations

from dataclasses import dataclass

from proxy_client import ProxyClient
from e2e_http import StreamingResponse
from models import (
    ChatBody,
    ChatMessage,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
)

MODEL_ACCESS_DENIED_MARKER = "key_model_access_denied"
ROUTE_NOT_ALLOWED_MARKER = "not allowed to call this route"


@dataclass(frozen=True, slots=True)
class AccessControlClient:
    proxy: ProxyClient

    def llm_only_key(self) -> str:
        return self.proxy.generate_key(
            KeyGenerateBody(models=[], allowed_routes=["llm_api_routes"])
        )

    def delete_key(self, key: str) -> None:
        self.proxy.delete_key(key)

    def chat_status(self, key: str, model: str, content: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model, messages=[ChatMessage(role="user", content=content)]
            ),
        )

    def create_model_status(self, key: str, model_name: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/model/new",
            headers=self.proxy.transport.bearer(key),
            json=ModelNewBody(
                model_name=model_name,
                litellm_params=LiteLLMParamsBody(model="openai/gpt-4o-mini"),
                model_info=ModelInfoBody(id=model_name),
            ),
        )


def build_client(proxy: ProxyClient) -> AccessControlClient:
    return AccessControlClient(proxy=proxy)
