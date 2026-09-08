from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Optional

import httpx
from httpx import Response
from pydantic import BaseModel, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.base_llm.passthrough.transformation import (
    BasePassthroughConfig,
    relayed_json_object,
    replace_path_segment,
    strip_leading_model_segment,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CallTypes, EmbeddingResponse, ImageResponse

if TYPE_CHECKING:
    from httpx import URL

    from litellm.types.utils import CostResponseTypes


class RelayedChatRequest(BaseModel):
    messages: Sequence[Mapping[str, object]] | None = None


class RelayedCallDetails(BaseModel):
    request_data: RelayedChatRequest | None = None


def _relayed_messages(litellm_logging_obj: Logging) -> Sequence[Mapping[str, object]] | None:
    try:
        details: Final = RelayedCallDetails.model_validate(litellm_logging_obj.model_call_details)
    except ValidationError:
        return None
    return details.request_data.messages if details.request_data else None


@dataclass(frozen=True, slots=True)
class OpenAIRelayShape:
    path_suffix: str
    call_type: CallTypes
    parse: Callable[[Mapping[str, object]], EmbeddingResponse | ImageResponse | ResponsesAPIResponse]


OPENAI_RELAY_SHAPES: Final = (
    OpenAIRelayShape("/embeddings", CallTypes.aembedding, EmbeddingResponse.model_validate),
    OpenAIRelayShape("/responses", CallTypes.aresponses, ResponsesAPIResponse.model_validate),
    OpenAIRelayShape("/images/generations", CallTypes.aimage_generation, ImageResponse.model_validate),
)


def logged_openai_response(
    httpx_response: Response, logging_obj: Logging, endpoint: str
) -> EmbeddingResponse | ImageResponse | ResponsesAPIResponse | None:
    relayed_path: Final = f"/{endpoint.strip('/')}"
    shape: Final = next(
        (candidate for candidate in OPENAI_RELAY_SHAPES if relayed_path.endswith(candidate.path_suffix)), None
    )
    body: Final = relayed_json_object(httpx_response) if shape else None
    if shape is None or body is None:
        return None
    try:
        parsed: Final = shape.parse(body)
    except ValidationError:
        return None
    logging_obj.call_type = (
        shape.call_type.value
    )  # rebind-ok: routes cost calculation to the relayed shape's pricing path
    return parsed


class AzurePassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return bool(request_data.get("stream"))

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict | None,
        litellm_params: dict,
    ) -> tuple["URL", str]:
        base_target_url: Final = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("Azure api base not found")

        litellm_metadata: Final = litellm_params.get("litellm_metadata") or {}
        model_group: Final = litellm_metadata.get("model_group")
        routed_endpoint: Final = replace_path_segment(endpoint, model_group, model) if model_group else endpoint
        native_endpoint: Final = strip_leading_model_segment(routed_endpoint, (model,))

        caller_api_version: Final = request_query_params.get("api-version") if request_query_params else None
        complete_url: Final = BaseAzureLLM._get_base_azure_url(
            api_base=base_target_url,
            litellm_params={**litellm_params, "api_version": caller_api_version or litellm_params.get("api_version")},
            route=native_endpoint,
        )
        return (
            httpx.URL(complete_url),
            base_target_url,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers,
            litellm_params=GenericLiteLLMParams(**{**litellm_params, "api_key": api_key}),
        )

    @staticmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str | None:
        return api_base or get_secret_str("AZURE_API_BASE")

    @staticmethod
    def get_api_key(
        api_key: str | None = None,
    ) -> str | None:
        return api_key or get_secret_str("AZURE_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return super().get_models(api_key, api_base)

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: dict,
        logging_obj: Logging,
        endpoint: str,
    ) -> Optional["CostResponseTypes | ResponsesAPIResponse"]:
        from litellm import encoding
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        from litellm.types.utils import ModelResponse

        if "chat/completions" not in endpoint:
            return logged_openai_response(httpx_response, logging_obj, endpoint)

        openai_chat_config: Final = OpenAIGPTConfig()

        litellm_model_response: Final[ModelResponse] = openai_chat_config.transform_response(
            model=model,
            messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
            raw_response=httpx_response,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            optional_params={},
            litellm_params={},
            api_key="",
            request_data=request_data,
            encoding=encoding,
        )

        return litellm_model_response

    def handle_logging_collected_chunks(
        self,
        all_chunks: Sequence[str],
        litellm_logging_obj: Logging,
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        if "chat/completions" not in endpoint:
            return None

        return OpenAIPassthroughLoggingHandler()._build_complete_streaming_response(  # pyright: ignore[reportPrivateUsage]  # the only OpenAI SSE-to-ModelResponse assembler; reimplementing it would fork the parser
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            messages=_relayed_messages(litellm_logging_obj),
        )
