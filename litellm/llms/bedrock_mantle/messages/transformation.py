from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import MANTLE_MESSAGES_PATH
from litellm.llms.bedrock.messages.mantle_transformation import AmazonMantleMessagesConfig
from litellm.llms.bedrock_mantle.common_utils import (
    MANTLE_HOST_RE,
    BedrockMantleAuthMixin,
    resolve_mantle_region,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES

_BASE_SUFFIXES_TO_STRIP: Final = (
    MANTLE_MESSAGES_PATH,
    "/v1/messages",
    "/messages",
    "/anthropic/v1",
    "/openai/v1",
    "/v1",
)


def build_mantle_native_messages_url(api_base: str | None, litellm_params: Mapping[str, object]) -> str:
    region: Final = resolve_mantle_region(MappingProxyType({**litellm_params, "api_base": api_base}))
    configured: Final = (
        api_base or get_secret_str("BEDROCK_MANTLE_API_BASE") or f"https://bedrock-mantle.{region}.api.aws"
    ).rstrip("/")
    stripped: Final = next(
        (configured[: -len(suffix)] for suffix in _BASE_SUFFIXES_TO_STRIP if configured.endswith(suffix)),
        configured,
    )
    host: Final = f"https://bedrock-mantle.{region}.api.aws" if MANTLE_HOST_RE.match(stripped) else stripped
    return f"{host}{MANTLE_MESSAGES_PATH}"


class BedrockMantleAnthropicMessagesConfig(BedrockMantleAuthMixin, AmazonMantleMessagesConfig):
    _BEDROCK_INVOKE_SUPPORTED_CONTEXT_MANAGEMENT_EDITS: Mapping[str, str] = MappingProxyType(
        {
            **AmazonMantleMessagesConfig._BEDROCK_INVOKE_SUPPORTED_CONTEXT_MANAGEMENT_EDITS,
            "clear_thinking_20251015": ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value,
        }
    )

    def __init__(self, aws_signer: BaseAWSLLM | None = None) -> None:
        AmazonMantleMessagesConfig.__init__(self)
        self._aws_signer = aws_signer or self

    @property
    def custom_llm_provider(self) -> str | None:
        return "bedrock_mantle"

    def uses_get_llm_provider_api_base(self) -> bool:
        return True

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return build_mantle_native_messages_url(api_base=api_base, litellm_params=litellm_params)
