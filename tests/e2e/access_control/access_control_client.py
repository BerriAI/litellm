"""Client for the access-control e2e suite."""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway
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
    gateway: Gateway

    def llm_only_key(self) -> str:
        return self.gateway.generate_key(
            KeyGenerateBody(models=[], allowed_routes=["llm_api_routes"])
        )

    def delete_key(self, key: str) -> None:
        self.gateway.delete_key(key)

    def chat_status(self, key: str, model: str, content: str) -> StreamingResponse:
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=ChatBody(
                model=model, messages=[ChatMessage(role="user", content=content)]
            ),
        )

    def create_model_status(self, key: str, model_name: str) -> StreamingResponse:
        return self.gateway.transport.send(
            "/model/new",
            headers=self.gateway.transport.bearer(key),
            json=ModelNewBody(
                model_name=model_name,
                litellm_params=LiteLLMParamsBody(model="openai/gpt-4o-mini"),
                model_info=ModelInfoBody(id=model_name),
            ),
        )


def build_client() -> AccessControlClient:
    return AccessControlClient(gateway=build_gateway())
