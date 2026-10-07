from dataclasses import dataclass
from typing import Final


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
        return model

    def request_model(self, model: str) -> str:
        return model

    def endpoint_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/')}/{model}/invocations"

    def unwrap_response(self, payload: object) -> object:
        return payload


DATABRICKS_DECISIONS_ENDPOINT: Final[DatabricksDecisionsEndpoint] = DatabricksDecisionsEndpoint()
