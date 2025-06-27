from typing import TYPE_CHECKING, Optional, Tuple

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockModelInfo

if TYPE_CHECKING:
    from httpx import URL


class BedrockPassthroughConfig(BaseAWSLLM, BedrockModelInfo, BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return "stream" in endpoint

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        optional_params = litellm_params.copy()

        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params,
            model=model,
            model_id=None,
        )

        api_base = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"

        return self.format_url(endpoint, api_base, request_query_params or {}), api_base

    def sign_request(
        self,
        headers: dict,
        litellm_params: dict,
        request_data: Optional[dict],
        api_base: str,
        model: Optional[str] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        optional_params = litellm_params.copy()
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data or {},
            api_base=api_base,
            model=model,
        )
