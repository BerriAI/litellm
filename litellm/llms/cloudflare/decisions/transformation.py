from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.secret_managers.main import (
    get_secret_str,
    normalize_nonempty_secret_str,
)

_RESPONSE_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


class CloudflareDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("CLOUDFLARE_API_KEY",)
    api_base_env = ("CLOUDFLARE_API_BASE",)

    def get_default_api_base(self) -> str | None:
        account_id: Final = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
        if account_id is None:
            return None
        return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID or pass api_base explicitly"

    def canonical_model(self, model: str) -> str:
        if model.startswith("@cf/"):
            return model
        return f"@cf/cloudflare/{model}"

    def supports_audio_video_input(self, model: str) -> bool:
        return self.canonical_model(model) == "@cf/cloudflare/clef-omni"

    def request_model(self, model: str) -> str:
        return model.rsplit("/", maxsplit=1)[-1]

    def get_complete_url(self, api_base: str, model: str) -> str:
        normalized_api_base: Final = api_base.rstrip("/")
        if normalized_api_base.endswith("/ai/v1"):
            return f"{normalized_api_base.removesuffix('/ai/v1')}/ai/run/{model}"
        if normalized_api_base.endswith("/ai/run"):
            return f"{normalized_api_base}/{model}"
        return f"{normalized_api_base}/ai/run/{model}"

    def unwrap_response(self, payload: object) -> object:
        try:
            response_mapping: Final = _RESPONSE_MAPPING_ADAPTER.validate_python(payload)
        except ValidationError:
            return payload
        if "answers" in response_mapping:
            return response_mapping
        try:
            return _RESPONSE_MAPPING_ADAPTER.validate_python(response_mapping.get("result"))
        except ValidationError:
            return response_mapping
