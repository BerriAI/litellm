"""
Support for Huawei Cloud ModelArts MaaS (Model as a Service) chat completions endpoint.

OpenAI-compatible API:
  https://api-ap-southeast-1.modelarts-maas.com/openai/v1/chat/completions

MaaS Standard API v1:
  https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions

Both APIs use Bearer token authentication:
  Authorization: Bearer <HUAWEI_CLOUD_API_KEY>

Docs:
  https://support.huaweicloud.com/intl/en-us/usermanual-maas/usermanual_maas_0022.html
  https://support.huaweicloud.com/intl/en-us/usermanual-maas-modelarts/maas-modelarts-0078.html
"""

from typing import List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.huawei_cloud.common_utils import HuaweiCloudException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


_DEFAULT_API_BASE = "https://api-ap-southeast-1.modelarts-maas.com/openai/v1"


class HuaweiCloudChatConfig(OpenAIGPTConfig):
    """
    Configuration for Huawei Cloud ModelArts MaaS chat completions.

    The OpenAI-compatible endpoint is used by default. Set the
    ``HUAWEI_CLOUD_API_BASE`` environment variable (or pass ``api_base``) to
    switch to a different regional endpoint or to the MaaS Standard API v1.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "huawei_cloud"

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        resolved_base = (
            api_base
            or get_secret_str("HUAWEI_CLOUD_API_BASE")
            or _DEFAULT_API_BASE
        )
        resolved_key = api_key or get_secret_str("HUAWEI_CLOUD_API_KEY")
        return resolved_base, resolved_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base, _ = self._get_openai_compatible_provider_info(api_base, api_key)
        base = (base or _DEFAULT_API_BASE).rstrip("/")
        if not base.endswith("/chat/completions"):
            base = f"{base}/chat/completions"
        return base

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return HuaweiCloudException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        extra_body = optional_params.pop("extra_body", {})
        response = super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
        response.update(extra_body)
        return response
