from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from httpx import Response

from litellm.constants import PLACEHOLDER_API_KEY
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.openai_like.passthrough.transformation import (
    logged_openai_like_response,
    logged_openai_like_stream,
    logged_text_completion_response,
    logged_text_completion_stream,
    relays_text_completion,
)
from litellm.secret_managers.main import (
    GOOGLE_OIDC_SECRET_PREFIX,
    UnresolvableSecretReference,
    get_secret_str,
    resolve_local_oidc_reference,
)

from ..common_utils import VLLMModelInfo, endpoint_under_api_base

if TYPE_CHECKING:
    from httpx import URL

    from litellm.llms.base_llm.passthrough.transformation import LoggedRelayResponse


class VLLMPassthroughConfig(VLLMModelInfo, BasePassthroughConfig):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return None if api_key == PLACEHOLDER_API_KEY else api_key

    def upstream_api_key(self, api_key: str | None, api_base: str | None) -> str | None:
        if api_key is not None or not self._is_vllm_api_base(api_base):
            return api_key
        vllm_api_key: Final = get_secret_str("VLLM_API_KEY")
        if vllm_api_key is None or vllm_api_key.startswith(GOOGLE_OIDC_SECRET_PREFIX):
            return None
        match resolve_local_oidc_reference(vllm_api_key):
            case UnresolvableSecretReference():
                return None
            case str() as resolved:
                return resolved

    @staticmethod
    def _is_vllm_api_base(api_base: str | None) -> bool:
        vllm_api_base: Final = get_secret_str("VLLM_API_BASE")
        return api_base is not None and vllm_api_base is not None and api_base.rstrip("/") == vllm_api_base.rstrip("/")

    def upstream_auth_headers(self, api_key: str) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

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
            self.format_url(endpoint_under_api_base(endpoint, base_target_url), base_target_url, request_query_params),
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
        return logged_openai_like_response(model, httpx_response, request_data, logging_obj, endpoint)

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
