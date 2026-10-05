from dataclasses import dataclass
from typing import Protocol


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

    def unwrap_response(self, payload: object) -> object: ...
