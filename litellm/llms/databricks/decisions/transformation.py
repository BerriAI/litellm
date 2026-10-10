import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig

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


class DatabricksDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("DATABRICKS_API_KEY", "DATABRICKS_TOKEN")
    api_base_env = ("DATABRICKS_API_BASE",)

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return (
            f"api_base is required for Decisions provider '{custom_llm_provider}': set DATABRICKS_API_BASE to "
            "https://<workspace-host>/serving-endpoints"
        )

    def canonical_model(self, model: str) -> str:
        return validate_serving_endpoint_name(model)

    def get_complete_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/')}/{validate_serving_endpoint_name(model)}/invocations"

    def classifier_response(self, body: Mapping[str, object], requested_model: str) -> Mapping[str, object]:
        return MappingProxyType({**body, "model": requested_model})

    def connection(self, api_base: str | None, api_key: str | None) -> DatabricksDecisionsConnection:
        base: Final = self.resolve_api_base(api_base)
        key: Final = api_key if api_base is not None else self.resolve_api_key(api_key)
        if not base or not key:
            raise ValueError(
                "Databricks requires api_key or DATABRICKS_API_KEY (or DATABRICKS_TOKEN) and api_base or "
                "DATABRICKS_API_BASE pointing to https://<workspace-host>/serving-endpoints"
            )
        return DatabricksDecisionsConnection(api_base=base.rstrip("/"), api_key=key)


DATABRICKS_DECISIONS_CONFIG: Final[DatabricksDecisionsConfig] = DatabricksDecisionsConfig()
