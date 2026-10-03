from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

from litellm.secret_managers.main import (
    get_secret_str,
    normalize_nonempty_secret_str,
)

_RESPONSE_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class CloudflareDecisionsEndpoint:
    api_key_env: tuple[str, ...] = ("CLOUDFLARE_API_KEY",)
    api_base_env: str = "CLOUDFLARE_API_BASE"
    api_key_required: bool = True

    def default_api_base(self) -> str | None:
        account_id: Final = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
        if account_id is None:
            return None
        return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"

    def missing_api_base_message(self, provider: str) -> str:
        return "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID or pass api_base explicitly"

    def canonical_model(self, model: str) -> str:
        if model.startswith("@cf/"):
            return model
        return f"@cf/cloudflare/{model}"

    def request_model(self, model: str) -> str:
        return model.rsplit("/", maxsplit=1)[-1]

    def endpoint_url(self, api_base: str, model: str) -> str:
        normalized_api_base: Final = api_base.rstrip("/")
        if normalized_api_base.endswith("/ai/v1"):
            return f"{normalized_api_base.removesuffix('/ai/v1')}/ai/run/{model}"
        if normalized_api_base.endswith("/ai/run"):
            return f"{normalized_api_base}/{model}"
        return f"{normalized_api_base}/ai/run/{model}"

    def unwrap_response(self, payload: object) -> object:
        if not isinstance(payload, Mapping):
            return payload
        response_mapping: Final = _RESPONSE_MAPPING_ADAPTER.validate_python(payload)
        if "answers" in response_mapping:
            return payload
        result: Final = response_mapping.get("result")
        if isinstance(result, Mapping):
            return result
        return payload


CLOUDFLARE_DECISIONS_ENDPOINT: Final[CloudflareDecisionsEndpoint] = CloudflareDecisionsEndpoint()
