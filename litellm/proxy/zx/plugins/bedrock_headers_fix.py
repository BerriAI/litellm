from typing import Any, Literal

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth
from litellm.types.utils import (
    CallTypesLiteral,
)

class BedrockHeadersFix(CustomLogger):
    """Fix Bedrock request headers

    Custom callback to strip headers that may cause problems with upstream requests to Amazon Bedrock when
    forward_client_headers_to_llm_api is enabled.
    """

    FORBIDDEN_HEADERS = (
        "x-amzn-tls-version",
        "x-amzn-tls-cipher-suite",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-request-start",
    )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, Any],
        call_type: CallTypesLiteral,
    ) -> dict[str, Any]:
        if "headers" not in data:
            return data
        return {
            **data,
            "headers": {
                k: v for k, v in data["headers"].items() if k not in self.FORBIDDEN_HEADERS
            },
        }


proxy_handler_instance = BedrockHeadersFix()