from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from litellm.types.decisions import DecisionsRequestBody


def systemone_request_body(model: str, request: DecisionsRequestBody) -> Mapping[str, object]:
    return {
        "model": model,
        "state": request.state,
        "questions": {
            name: question.model_dump(mode="json", exclude_none=True) for name, question in request.questions.items()
        },
    }


@dataclass(frozen=True, slots=True)
class JevCompatibleDecisionsEndpoint:
    default_api_base_value: str | None
    path: str
    api_key_env: tuple[str, ...]
    api_base_env: str
    api_key_required: bool = True

    def default_api_base(self) -> str | None:
        return self.default_api_base_value

    def missing_api_base_message(self, provider: str) -> str:
        return f"api_base is required for Decisions provider '{provider}'"

    def canonical_model(self, model: str) -> str:
        return model

    def request_model(self, model: str) -> str:
        return model

    def endpoint_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1')}{self.path}"

    def request_body(self, model: str, request: DecisionsRequestBody) -> Mapping[str, object]:
        return systemone_request_body(model, request)

    def unwrap_response(self, payload: object) -> object:
        return payload


class DecisionsProviderConfig(Protocol):
    @property
    def api_key_env(self) -> tuple[str, ...]: ...

    @property
    def api_base_env(self) -> str: ...

    @property
    def api_key_required(self) -> bool: ...

    def default_api_base(self) -> str | None: ...

    def missing_api_base_message(self, provider: str) -> str: ...

    def canonical_model(self, model: str) -> str: ...

    def request_model(self, model: str) -> str: ...

    def endpoint_url(self, api_base: str, model: str) -> str: ...

    def request_body(self, model: str, request: DecisionsRequestBody) -> Mapping[str, object]: ...

    def unwrap_response(self, payload: object) -> object: ...
