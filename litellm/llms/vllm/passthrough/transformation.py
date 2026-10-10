from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from httpx import Response

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.openai_like.passthrough.transformation import logged_openai_like_response, logged_openai_like_stream

from ..common_utils import VLLMModelInfo

if TYPE_CHECKING:
    from httpx import URL

    from litellm.llms.base_llm.passthrough.transformation import LoggedRelayResponse


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
        return logged_openai_like_response(model, httpx_response, request_data, logging_obj, endpoint)

    def handle_logging_collected_chunks(
        self,
        all_chunks: Sequence[str],
        litellm_logging_obj: Logging,
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> "LoggedRelayResponse | None":
        return logged_openai_like_stream(all_chunks, litellm_logging_obj, model, endpoint)
