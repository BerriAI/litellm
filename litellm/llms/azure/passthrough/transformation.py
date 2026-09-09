import re
from collections.abc import Callable, Collection, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional

import httpx
from httpx import Response
from pydantic import BaseModel, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.base_llm.passthrough.transformation import (
    BasePassthroughConfig,
    RelayShape,
    logged_relay_shape,
    replace_path_segment,
    strip_leading_model_segment,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ResponsesAPIResponse, ResponsesTerminalEvent
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CallTypes, EmbeddingResponse, ImageResponse

if TYPE_CHECKING:
    from httpx import URL

    from litellm.llms.base_llm.passthrough.transformation import LoggedRelayResponse


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


RESPONSES_RELAY_SHAPE: Final = RelayShape("/responses", CallTypes.aresponses, ResponsesAPIResponse.model_validate)

OPENAI_RELAY_SHAPES: Final = (
    RelayShape("/embeddings", CallTypes.aembedding, EmbeddingResponse.model_validate),
    RESPONSES_RELAY_SHAPE,
    RelayShape("/images/generations", CallTypes.aimage_generation, ImageResponse.model_validate),
)


def logged_responses_stream(all_chunks: Sequence[str], logging_obj: Logging) -> ResponsesTerminalEvent | None:
    """A streaming logging object assembles the logged response from the terminal event, not from its body."""
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    terminal_event: Final = OpenAIResponsesAPIConfig.parse_terminal_event_from_stream_chunks(all_chunks=all_chunks)
    if terminal_event is None:
        return None
    logging_obj.call_type = (
        RESPONSES_RELAY_SHAPE.call_type.value
    )  # rebind-ok: routes cost calculation to the relayed shape's pricing path
    return terminal_event


AZURE_DEPLOYMENT_SEGMENT: Final = re.compile(r"(?<![^/])openai/deployments/([^/]+)")


def azure_router_model_in_endpoint(endpoint: str, router_models: Collection[str]) -> str | None:
    parts: Final = endpoint.split("/")
    if len(parts) < 2:
        return None
    return next((part for part in parts if part in router_models), None)


def foreign_azure_deployment(
    endpoint: str, model_group: str, served_models: Callable[[], Collection[str]]
) -> str | None:
    match: Final = AZURE_DEPLOYMENT_SEGMENT.search(endpoint)
    if match is None:
        return None
    deployment: Final = match.group(1)
    if deployment == model_group:
        return None
    served: Final = frozenset(name.casefold() for name in served_models())
    return None if deployment.casefold() in served else deployment


def without_api_version(api_base: str) -> str:
    url: Final = httpx.URL(api_base)
    kept_params: Final = tuple((key, value) for key, value in url.params.multi_items() if key != "api-version")
    return str(url.copy_with(params=httpx.QueryParams(kept_params)))


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
        relay_base: Final = without_api_version(base_target_url) if caller_api_version else base_target_url
        complete_url: Final = BaseAzureLLM._get_base_azure_url(
            api_base=relay_base,
            litellm_params=MappingProxyType(
                {**litellm_params, "api_version": caller_api_version or litellm_params.get("api_version")}
            ),
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
    ) -> Optional["LoggedRelayResponse"]:
        from litellm import encoding
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        from litellm.types.utils import ModelResponse

        if "chat/completions" not in endpoint:
            return logged_relay_shape(OPENAI_RELAY_SHAPES, httpx_response, logging_obj, endpoint)

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
    ) -> Optional["LoggedRelayResponse"]:
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        if f"/{endpoint.strip('/')}".endswith(RESPONSES_RELAY_SHAPE.path_suffix):
            return logged_responses_stream(all_chunks, litellm_logging_obj)
        if "chat/completions" not in endpoint:
            return None

        return OpenAIPassthroughLoggingHandler()._build_complete_streaming_response(  # pyright: ignore[reportPrivateUsage]  # the only OpenAI SSE-to-ModelResponse assembler; reimplementing it would fork the parser
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            messages=_relayed_messages(litellm_logging_obj),
        )
