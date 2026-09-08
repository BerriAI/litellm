from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote

import httpx
from openai.types.responses import EasyInputMessageParam, ResponseInputItemParam
from pydantic import TypeAdapter

from litellm.llms.fireworks_ai.common_utils import (
    resolve_fireworks_api_key,
    resolve_fireworks_resource_name,
    with_fireworks_session_affinity,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

FIREWORKS_AI_DEFAULT_API_BASE: Final = "https://api.fireworks.ai/inference/v1"


def _session_params(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    extras: Final[Mapping[str, object]] = litellm_params.model_extra or MappingProxyType({})
    return MappingProxyType(
        {"litellm_session_id": extras.get("litellm_session_id"), "metadata": extras.get("litellm_metadata")}
    )


_instructions_adapter: Final = TypeAdapter[str | None](str | None)


def _instruction_text(item: ResponseInputItemParam) -> str | None:
    if "role" not in item or (item["role"] != "system" and item["role"] != "developer"):
        return None
    content: Final = item["content"]
    if isinstance(content, str):
        return content
    return "\n\n".join(part["text"] for part in content if part["type"] == "input_text")


def _with_single_leading_system_item(
    input: str | ResponseInputParam, instructions: str | None
) -> str | ResponseInputParam:
    items: Final = () if isinstance(input, str) else tuple(input)
    instruction_texts: Final = tuple(text for text in (instructions, *map(_instruction_text, items)) if text)
    if not instruction_texts:
        return input
    leading: Final = EasyInputMessageParam(role="system", content="\n\n".join(instruction_texts), type="message")
    rest: Final = (
        (EasyInputMessageParam(role="user", content=input),)
        if isinstance(input, str)
        else tuple(item for item in items if _instruction_text(item) is None)
    )
    return [leading, *rest]


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
        return dict(pinned)  # mutable-ok: the HTTP handler updates the returned headers in place

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        base: Final = (api_base or get_secret_str("FIREWORKS_API_BASE") or FIREWORKS_AI_DEFAULT_API_BASE).rstrip("/")
        return f"{base}/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,  # mutable-ok: overrides the base class signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: overrides the base class signature
    ) -> dict:  # mutable-ok: overrides the base class signature
        instructions: Final = _instructions_adapter.validate_python(
            response_api_optional_request_params.get("instructions")
        )
        folded_params: Final = {  # mutable-ok: the base class takes the optional params as a dict
            key: value for key, value in response_api_optional_request_params.items() if key != "instructions"
        }
        return super().transform_responses_api_request(
            model=resolve_fireworks_resource_name(model),
            input=_with_single_leading_system_item(self._validate_input_param(input), instructions),
            response_api_optional_request_params=folded_params,
            litellm_params=litellm_params,
            headers=headers,
        )

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
