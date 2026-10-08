import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from litellm.secret_managers.main import get_secret_str

_SERVING_ENDPOINT_NAME: Final = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")


def validate_serving_endpoint_name(name: str) -> str:
    if _SERVING_ENDPOINT_NAME.fullmatch(name) is None:
        raise ValueError(
            f"Databricks serving endpoint name {name!r} must be the bare endpoint name (letters, digits, '-', '_' "
            "and '.', with no '/', '?', '#', spaces, or a leading '.'), e.g. databricks-openjev-qwen35-4b"
        )
    return name


@dataclass(frozen=True, slots=True)
class DatabricksDecisionsConnection:
    api_base: str
    api_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class DatabricksDecisionsEndpoint:
    api_key_env: tuple[str, ...] = ("DATABRICKS_API_KEY", "DATABRICKS_TOKEN")
    api_base_env: str = "DATABRICKS_API_BASE"
    api_key_required: bool = True

    def default_api_base(self) -> str | None:
        return None

    def missing_api_base_message(self, provider: str) -> str:
        return (
            f"api_base is required for Decisions provider '{provider}': set DATABRICKS_API_BASE to "
            "https://<workspace-host>/serving-endpoints"
        )

    def canonical_model(self, model: str) -> str:
        return validate_serving_endpoint_name(model)

    def request_model(self, model: str) -> str:
        return model

    def endpoint_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/')}/{validate_serving_endpoint_name(model)}/invocations"

    def unwrap_response(self, payload: object) -> object:
        return payload

    def classifier_response(self, body: Mapping[str, object], requested_model: str) -> Mapping[str, object]:
        return MappingProxyType({**body, "model": requested_model})

    def connection(self, api_base: str | None, api_key: str | None) -> DatabricksDecisionsConnection:
        base: Final = api_base or get_secret_str(self.api_base_env)
        key: Final = api_key if api_base is not None else api_key or self._environment_api_key()
        if not base or not key:
            raise ValueError(
                "Databricks requires api_key or DATABRICKS_API_KEY (or DATABRICKS_TOKEN) and api_base or "
                "DATABRICKS_API_BASE pointing to https://<workspace-host>/serving-endpoints"
            )
        return DatabricksDecisionsConnection(api_base=base.rstrip("/"), api_key=key)

    def _environment_api_key(self) -> str | None:
        return next((key for key in map(get_secret_str, self.api_key_env) if key), None)


DATABRICKS_DECISIONS_ENDPOINT: Final[DatabricksDecisionsEndpoint] = DatabricksDecisionsEndpoint()
