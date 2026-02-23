from typing import TYPE_CHECKING, Optional, Tuple

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

from ..common_utils import VLLMModelInfo

if TYPE_CHECKING:
    from httpx import URL


class VLLMPassthroughConfig(VLLMModelInfo, BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return "stream" in request_data

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        base_target_url = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("VLLM api base not found")

        return (
            self.format_url(endpoint, base_target_url, request_query_params),
            base_target_url,
        )
