from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote

import httpx

from litellm.llms.fireworks_ai.common_utils import (
    FIREWORKS_FORWARD_USER_ID_PARAM,
    get_fireworks_forwarded_user_id,
    resolve_fireworks_api_key,
    resolve_fireworks_resource_name,
    with_fireworks_session_affinity,
    without_caller_user,
)
from litellm.llms.openai.responses.instruction_items import developer_items_as_system, instruction_items_folded
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import declared_mid_conversation_system_support, deployment_model_info

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

FIREWORKS_AI_DEFAULT_API_BASE: Final = "https://api.fireworks.ai/inference/v1"


def _session_params(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    extras: Final[Mapping[str, object]] = litellm_params.model_extra or MappingProxyType({})
    return MappingProxyType(
        {"litellm_session_id": extras.get("litellm_session_id"), "metadata": extras.get("litellm_metadata")}
    )


def _forwarded_user_params(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    extras: Final[Mapping[str, object]] = litellm_params.model_extra or MappingProxyType({})
    return MappingProxyType(
        {
            FIREWORKS_FORWARD_USER_ID_PARAM: litellm_params.fireworks_forward_user_id,
            "litellm_metadata": extras.get("litellm_metadata"),
        }
    )


class FireworksAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.FIREWORKS_AI

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: overrides the base class signature
        params: Final = litellm_params or GenericLiteLLMParams()
        api_key: Final = resolve_fireworks_api_key(params.api_key)
        if api_key is None:
            raise ValueError("FIREWORKS_API_KEY is not set")
        authorized: Final = MappingProxyType(
            {"Content-Type": "application/json", **headers, "Authorization": f"Bearer {api_key}"}
        )
        pinned: Final = with_fireworks_session_affinity(authorized, _session_params(params))
        return dict(pinned)

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        base: Final = (api_base or get_secret_str("FIREWORKS_API_BASE") or FIREWORKS_AI_DEFAULT_API_BASE).rstrip("/")
        return f"{base}/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict[str, object],  # mutable-ok: overrides the base class signature
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: overrides the base class signature
    ) -> dict[str, object]:  # mutable-ok: overrides the base class signature
        validated_input: Final = self._validate_input_param(input)
        declared: Final = declared_mid_conversation_system_support(
            model, self.custom_llm_provider.value, deployment_model_info(litellm_params)
        )
        folded: Final = (
            None
            if declared is True
            else instruction_items_folded(
                validated_input, response_api_optional_request_params, "user" if declared is False else "system"
            )
        )
        placed_input, placed_params = folded or (
            developer_items_as_system(validated_input),
            response_api_optional_request_params,
        )
        request: Final = super().transform_responses_api_request(
            model=resolve_fireworks_resource_name(model),
            input=placed_input,
            response_api_optional_request_params=dict(placed_params),
            litellm_params=litellm_params,
            headers=headers,
        )
        forwarded_user_id: Final = get_fireworks_forwarded_user_id(_forwarded_user_params(litellm_params))
        return request if forwarded_user_id is None else {**request, "user": forwarded_user_id}

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        return without_caller_user(extra_body, get_fireworks_forwarded_user_id(_forwarded_user_params(litellm_params)))

    def transform_delete_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> DeleteResponseResult:
        deleted_id: Final = unquote(raw_response.request.url.path.rsplit("/", 1)[-1])
        return DeleteResponseResult(id=deleted_id, object="response", deleted=True)

    def supports_native_websocket(self) -> bool:
        return False

    def supports_native_file_search(self) -> bool:
        return False
