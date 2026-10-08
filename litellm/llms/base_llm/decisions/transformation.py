from abc import ABC
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.decisions import DecisionsRequest, DecisionsResponse

PAYLOAD_ADAPTER: Final[TypeAdapter[object]] = TypeAdapter(object)
_RESPONSE_ADAPTER: Final[TypeAdapter[DecisionsResponse]] = TypeAdapter(DecisionsResponse)
_RESERVED_HEADERS: Final[frozenset[str]] = frozenset({"authorization", "content-type"})


class BaseDecisionsConfig(ABC):
    path: str = "/v1/systemone"
    api_key_env: tuple[str, ...] = ()
    api_base_env: tuple[str, ...] = ()
    api_key_required: bool = True

    def get_default_api_base(self) -> str | None:
        return None

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return f"api_base is required for Decisions provider '{custom_llm_provider}'"

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return api_base or self._first_secret(self.api_base_env) or self.get_default_api_base()

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key or self._first_secret(self.api_key_env)

    @staticmethod
    def _first_secret(names: tuple[str, ...]) -> str | None:
        return next((value for value in (get_secret_str(name) for name in names) if value), None)

    def canonical_model(self, model: str) -> str:
        return model

    def request_model(self, model: str) -> str:
        return model

    def validate_environment(self, headers: Mapping[str, str], model: str, api_key: str | None) -> dict[str, str]:
        return {
            **{name: value for name, value in headers.items() if name.lower() not in _RESERVED_HEADERS},
            **({"Authorization": f"Bearer {api_key}"} if api_key is not None else {}),
            "Content-Type": "application/json",
        }

    def get_complete_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1')}{self.path}"

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsRequest,
        custom_llm_provider: str,
    ) -> dict[str, object]:
        return {
            "model": self.request_model(model),
            "state": request.state,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in request.questions.items()
            },
        }

    def unwrap_response(self, payload: object) -> object:
        return payload

    def transform_decisions_response(
        self,
        model: str,
        custom_llm_provider: str,
        raw_response: httpx.Response,
        request: DecisionsRequest,
    ) -> DecisionsResponse:
        payload: Final[object] = PAYLOAD_ADAPTER.validate_json(raw_response.content)
        try:
            response: Final = _RESPONSE_ADAPTER.validate_python(self.unwrap_response(payload))
        except ValidationError as error:
            raise BaseLLMException(
                status_code=500,
                message=f"Decisions provider '{custom_llm_provider}' returned an unexpected response: {error}",
            ) from error
        self.set_hidden_params(response, model, custom_llm_provider)
        return response

    @staticmethod
    def set_hidden_params(response: DecisionsResponse, model: str, custom_llm_provider: str) -> None:
        response.set_hidden_params(
            {
                "model": f"{custom_llm_provider}/{model}",
                "custom_llm_provider": custom_llm_provider,
                "provider_response_model": f"{custom_llm_provider}/{model}",
            }
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=headers)
