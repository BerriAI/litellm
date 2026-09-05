from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    api_key_header_for_base,
    get_azure_ai_auth_headers,
)
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig, strip_leading_model_segment
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardPassThroughResponseObject

if TYPE_CHECKING:
    from httpx import URL, Response

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CostResponseTypes


EMPTY_QUERY: Final[Mapping[str, object]] = MappingProxyType({})


class PassthroughMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_group: str = ""


def model_group_from(litellm_params: Mapping[str, object]) -> str:
    try:
        return PassthroughMetadata.model_validate(litellm_params.get("litellm_metadata")).model_group
    except ValidationError:
        return ""


def api_version_from(litellm_params: Mapping[str, object]) -> str | None:
    try:
        return TypeAdapter(str | None).validate_python(litellm_params.get("api_version"))
    except ValidationError:
        return None


def foundry_root(api_base: str) -> str:
    url: Final = httpx.URL(api_base)
    segments: Final = tuple(segment for segment in url.path.split("/") if segment)
    root_segments: Final = segments[: segments.index("models")] if "models" in segments else segments
    return str(url.copy_with(path="/" + "/".join(root_segments), query=None)).rstrip("/")


def relay_query_params(
    request_query_params: Mapping[str, object] | None,
    deployment_api_version: str | None,
    api_base: str,
) -> Mapping[str, object] | None:
    if request_query_params and "api-version" in request_query_params:
        return request_query_params
    api_version: Final = deployment_api_version or httpx.URL(api_base).params.get("api-version")
    if api_version is None:
        return request_query_params
    return MappingProxyType({**(request_query_params or EMPTY_QUERY), "api-version": api_version})


def relayed_body(httpx_response: Response) -> str | dict:
    try:
        body: Final[object] = httpx_response.json()
    except ValueError:
        return httpx_response.text
    return body if isinstance(body, dict) else httpx_response.text


class AzureAIPassthroughConfig(AzureFoundryModelInfo, BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: Mapping[str, object]) -> bool:
        return bool(request_data.get("stream"))

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: Mapping[str, object] | None,
        litellm_params: Mapping[str, object],
    ) -> tuple[URL, str]:
        base_target_url: Final = self.get_api_base(api_base)
        if base_target_url is None:
            raise ValueError("Azure AI api base not found: set `api_base` on the deployment or AZURE_AI_API_BASE")

        root: Final = foundry_root(base_target_url)
        native_endpoint: Final = strip_leading_model_segment(endpoint, (model, model_group_from(litellm_params)))
        query_params: Final = relay_query_params(
            request_query_params, api_version_from(litellm_params), base_target_url
        )
        return (self.format_url(native_endpoint, root, query_params), root)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base class contract returns dict for httpx
        auth_headers: Final = get_azure_ai_auth_headers(
            api_key=api_key,
            litellm_params=litellm_params,
            api_key_header=api_key_header_for_base(api_base),
        )
        return {**headers, **auth_headers}  # mutable-ok: base class contract returns dict for httpx

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: Mapping[str, object],
        logging_obj: Logging,
        endpoint: str,
    ) -> CostResponseTypes | StandardPassThroughResponseObject | None:
        from litellm.llms.azure.passthrough.transformation import AzurePassthroughConfig

        chat_result: Final = AzurePassthroughConfig().logging_non_streaming_response(  # pyright: ignore[reportUnknownMemberType]  # the Azure config still types request_data as a bare dict
            model=model,
            custom_llm_provider=custom_llm_provider,
            httpx_response=httpx_response,
            request_data=dict(request_data),  # mutable-ok: AzurePassthroughConfig wants a dict
            logging_obj=logging_obj,
            endpoint=endpoint,
        )
        if chat_result is not None:
            return chat_result
        return StandardPassThroughResponseObject(response=relayed_body(httpx_response))

    def handle_logging_collected_chunks(
        self,
        all_chunks: Sequence[str],
        litellm_logging_obj: Logging,
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> CostResponseTypes | None:
        from litellm.llms.azure.passthrough.transformation import AzurePassthroughConfig

        return AzurePassthroughConfig().handle_logging_collected_chunks(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            custom_llm_provider=custom_llm_provider,
            endpoint=endpoint,
        )
