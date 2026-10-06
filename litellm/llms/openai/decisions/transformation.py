from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import PAYLOAD_ADAPTER, BaseDecisionsConfig
from litellm.types.decisions import DecisionsRequest, DecisionsResponse

_RESPONSE_ADAPTER: Final[TypeAdapter[DecisionsResponse]] = TypeAdapter(DecisionsResponse)


class OpenAIDecisionsConfig(BaseDecisionsConfig):
    path = "/v1/decisions"
    api_key_env = ("OPENAI_API_KEY",)
    api_base_env = ("OPENAI_BASE_URL", "OPENAI_API_BASE")

    def get_default_api_base(self) -> str | None:
        return "https://api.openai.com"

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key or litellm.api_key or litellm.openai_key or self._first_secret(self.api_key_env)

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsRequest,
        custom_llm_provider: str,
    ) -> dict[str, object]:
        return {"model": model, **request.model_dump(mode="json", exclude_none=True, exclude={"model"})}

    def transform_decisions_response(
        self,
        model: str,
        custom_llm_provider: str,
        raw_response: httpx.Response,
        request: DecisionsRequest,
    ) -> DecisionsResponse:
        try:
            response: Final = _RESPONSE_ADAPTER.validate_python(PAYLOAD_ADAPTER.validate_json(raw_response.content))
        except ValidationError as error:
            raise BaseLLMException(
                status_code=500,
                message=f"Decisions provider '{custom_llm_provider}' returned an unexpected response: {error}",
            ) from error
        self.set_hidden_params(response, model, custom_llm_provider)
        return response
