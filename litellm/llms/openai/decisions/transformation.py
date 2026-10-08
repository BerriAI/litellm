from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm.decisions.openai_transformation import to_openai_request, to_systemone_response
from litellm.types.decisions import DecisionsRequestBody, DecisionsResponse


@dataclass(frozen=True, slots=True)
class OpenAIDecisionsEndpoint:
    api_key_env: tuple[str, ...] = ("OPENAI_API_KEY",)
    api_base_env: str = "OPENAI_BASE_URL"
    api_key_required: bool = True

    def default_api_base(self) -> str | None:
        return "https://api.openai.com"

    def missing_api_base_message(self, provider: str) -> str:
        return f"api_base is required for Decisions provider '{provider}'"

    def canonical_model(self, model: str) -> str:
        return model

    def request_model(self, model: str) -> str:
        return model

    def endpoint_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1')}/v1/decisions"

    def request_body(self, model: str, request: DecisionsRequestBody) -> Mapping[str, object]:
        return to_openai_request(model, request)

    def unwrap_response(self, payload: object) -> DecisionsResponse:
        return to_systemone_response(payload)


OPENAI_DECISIONS_ENDPOINT: Final[OpenAIDecisionsEndpoint] = OpenAIDecisionsEndpoint()
