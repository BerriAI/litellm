from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    api_key_header_for_base,
    get_azure_ai_auth_headers,
)
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from httpx import URL, Response

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CostResponseTypes


def strip_leading_model_segment(endpoint: str, model_names: tuple[str, ...]) -> str:
    path: Final = endpoint.lstrip("/")
    for model_name in model_names:
        if not model_name:
            continue
        if path == model_name:
            return ""
        if path.startswith(f"{model_name}/"):
            return path[len(model_name) + 1 :]
    return path


class PassthroughMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_group: str = ""


def model_group_from(litellm_params: Mapping[str, object]) -> str:
    try:
        return PassthroughMetadata.model_validate(litellm_params.get("litellm_metadata")).model_group
    except ValidationError:
        return ""


class AzureAIPassthroughConfig(AzureFoundryModelInfo, BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: Mapping[str, object]) -> bool:
        return request_data.get("stream") is True

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

        native_endpoint: Final = strip_leading_model_segment(endpoint, (model, model_group_from(litellm_params)))
        return (
            self.format_url(native_endpoint, base_target_url, request_query_params),
            base_target_url,
        )

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
    ) -> CostResponseTypes | None:
        from litellm.llms.azure.passthrough.transformation import AzurePassthroughConfig

        return AzurePassthroughConfig().logging_non_streaming_response(  # pyright: ignore[reportUnknownMemberType]  # the Azure config still types request_data as a bare dict
            model=model,
            custom_llm_provider=custom_llm_provider,
            httpx_response=httpx_response,
            request_data=dict(request_data),  # mutable-ok: AzurePassthroughConfig wants a dict
            logging_obj=logging_obj,
            endpoint=endpoint,
        )
