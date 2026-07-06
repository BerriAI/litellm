"""Client for the model-management e2e suite: the shared Gateway plus the
/model/update write, a strict /model/delete, a /model/info lookup by name, and
the raw-status calls the tests judge by HTTP outcome (a forbidden /model/new, a
chat against a deleted model).
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, StreamingResponse, unwrap
from models import (
    ChatBody,
    ChatMessage,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelDeleteBody,
    ModelInfoBody,
    ModelInfoEntry,
    ModelNewBody,
    ModelUpdateBody,
    ModelUpdateParams,
)

ROUTE_NOT_ALLOWED_MARKER = "not allowed to call this route"
UNKNOWN_MODEL_MARKER = "Invalid model name passed in model="
NO_DEPLOYMENTS_MARKER = "There are no healthy deployments"


def is_deleted_model_rejection(outcome: StreamingResponse, model: str) -> bool:
    """True if the gateway refused the call because `model` is gone: a 400 naming
    the model, either the proxy's unknown-model shape (the data plane never knew
    the group) or the router's no-healthy-deployments shape (the group name
    outlives its last deployment in the router until restart)."""
    if outcome.status_code != 400 or model not in outcome.body:
        return False
    return UNKNOWN_MODEL_MARKER in outcome.body or NO_DEPLOYMENTS_MARKER in outcome.body


@dataclass(frozen=True, slots=True)
class ModelsMgmtClient:
    gateway: Gateway

    def llm_only_key(self) -> str:
        return self.gateway.generate_key(
            KeyGenerateBody(models=[], allowed_routes=["llm_api_routes"])
        )

    def find_deployment(self, model_name: str) -> ModelInfoEntry | None:
        return next(
            (entry for entry in self.gateway.model_info() if entry.model_name == model_name),
            None,
        )

    def update_model_tpm(self, model_id: str, tpm: int) -> None:
        _ = unwrap(
            self.gateway.transport.post(
                "/model/update",
                headers=self.gateway.transport.master,
                json=ModelUpdateBody(
                    litellm_params=ModelUpdateParams(tpm=tpm),
                    model_info=ModelInfoBody(id=model_id),
                ),
                response_type=NoBody,
            )
        )

    def delete_model(self, model_id: str) -> None:
        """Strict delete for the act phase of a test: a failed delete is a hard
        failure, unlike the warn-only Gateway.delete_model used at teardown."""
        _ = unwrap(
            self.gateway.transport.post(
                "/model/delete",
                headers=self.gateway.transport.master,
                json=ModelDeleteBody(id=model_id),
                response_type=NoBody,
            )
        )

    def create_model_status(
        self, key: str, model_name: str, litellm_params: LiteLLMParamsBody
    ) -> StreamingResponse:
        return self.gateway.transport.send(
            "/model/new",
            headers=self.gateway.transport.bearer(key),
            json=ModelNewBody(
                model_name=model_name,
                litellm_params=litellm_params,
                model_info=ModelInfoBody(id=model_name),
            ),
        )

    def chat_status(self, key: str, model: str, content: str) -> StreamingResponse:
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=16,
            ),
        )


def build_client() -> ModelsMgmtClient:
    return ModelsMgmtClient(gateway=build_gateway())
