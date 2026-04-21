"""
Default api_base / api_key for xAI when using the OpenAI SDK image endpoints.

xAI exposes /v1/images/generations in OpenAI-compatible form; callers use
openai_chat_completions.image_generation with the same credential defaults as chat.
"""

from typing import Callable, Dict, Optional, Tuple

from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.types.utils import LlmProviders


def _resolve_xai_openai_sdk_image_credentials(
    api_base: Optional[str],
    api_key: Optional[str],
    dynamic_api_key: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    api_base_resolved, key = XAIChatConfig()._get_openai_compatible_provider_info(
        api_base, api_key or dynamic_api_key
    )
    return api_base_resolved, key, None


OpenaiSdkImageCredentialResolver = Callable[
    [Optional[str], Optional[str], Optional[str]],
    Tuple[Optional[str], Optional[str], Optional[str]],
]

# Providers that route image_generation through the OpenAI SDK but need
# provider-specific defaults for api_base / api_key (extend here or in future
# neutral registry as more providers are added).
_OPENAI_SDK_IMAGE_CREDENTIAL_RESOLVERS: Dict[str, OpenaiSdkImageCredentialResolver] = {
    LlmProviders.XAI.value: _resolve_xai_openai_sdk_image_credentials,
}


def resolve_for_openai_sdk_image_generation(
    custom_llm_provider: str,
    api_base: Optional[str],
    api_key: Optional[str],
    dynamic_api_key: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Apply provider defaults before delegating to OpenAI SDK image APIs.

    Returns:
        (api_base, api_key, dynamic_api_key) possibly updated; dynamic_api_key may be
        cleared when the resolved key is folded into api_key for the downstream call.
    """
    resolver = _OPENAI_SDK_IMAGE_CREDENTIAL_RESOLVERS.get(custom_llm_provider)
    if resolver is None:
        return api_base, api_key, dynamic_api_key
    return resolver(api_base, api_key, dynamic_api_key)
