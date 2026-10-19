from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from httpx import Response
from pydantic import ConfigDict, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig, relayed_json_object
from litellm.llms.openai_like.passthrough.transformation import (
    logged_openai_like_response,
    logged_openai_like_stream,
    logged_text_completion_response,
    logged_text_completion_stream,
    relays_text_completion,
)
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import VLLMModelInfo

if TYPE_CHECKING:
    from httpx import URL

    from litellm.llms.base_llm.passthrough.transformation import LoggedRelayResponse


class ReportedUsage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class UsageReportingBody(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = None
    model: str | None = None
    usage: ReportedUsage


def logged_reported_usage(model: str, httpx_response: Response) -> ModelResponse | None:
    """
    The usage of a relayed body that has no OpenAI shape but reports one anyway.

    vLLM's own endpoints such as `/score`, `/classify` and `/pooling`, and servers built on vLLM such as
    vLLM Semantic Router's `/v1/systemone`, answer with a `usage` object (`prompt_tokens`/`completion_tokens`,
    or `input_tokens`/`output_tokens`) but none of the shapes the OpenAI relay parses. Without this, those
    calls are logged with no tokens at all.
    """
    body: Final = relayed_json_object(httpx_response)
    if body is None:
        return None
    try:
        relayed: Final = UsageReportingBody.model_validate(body)
    except ValidationError:
        return None
    usage: Final = relayed.usage
    prompt_tokens: Final = usage.prompt_tokens if usage.prompt_tokens is not None else usage.input_tokens
    completion_tokens: Final = usage.completion_tokens if usage.completion_tokens is not None else usage.output_tokens
    if prompt_tokens is None and completion_tokens is None:
        return None
    response: Final = ModelResponse(
        model=relayed.model or model,
        usage=Usage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
        ),
    )
    if relayed.id:
        response.id = relayed.id
    return response


class VLLMPassthroughConfig(VLLMModelInfo, BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return "stream" in request_data

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
            raise Exception("VLLM api base not found")

        return (
            self.format_url(endpoint, base_target_url, request_query_params),
            base_target_url,
        )

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: Mapping[str, object],
        logging_obj: Logging,
        endpoint: str,
    ) -> "LoggedRelayResponse | None":
        if relays_text_completion(endpoint):
            return logged_text_completion_response(httpx_response, logging_obj, endpoint)
        openai_shaped: Final = logged_openai_like_response(model, httpx_response, request_data, logging_obj, endpoint)
        if openai_shaped is not None:
            return openai_shaped
        return logged_reported_usage(model, httpx_response)

    def handle_logging_collected_chunks(
        self,
        all_chunks: Sequence[str],
        litellm_logging_obj: Logging,
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> "LoggedRelayResponse | None":
        if relays_text_completion(endpoint):
            return logged_text_completion_stream(all_chunks, litellm_logging_obj, model)
        return logged_openai_like_stream(all_chunks, litellm_logging_obj, model, endpoint)
