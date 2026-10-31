from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.common_utils import drop_clear_thinking_edits_for_model
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
    return join_api_base(host, MANTLE_MESSAGES_PATH)


class BedrockMantleAnthropicMessagesConfig(BedrockMantleAuthMixin, AmazonMantleMessagesConfig):
    _BEDROCK_INVOKE_SUPPORTED_CONTEXT_MANAGEMENT_EDITS: Mapping[str, str] = MappingProxyType(
        {
            **AmazonMantleMessagesConfig._BEDROCK_INVOKE_SUPPORTED_CONTEXT_MANAGEMENT_EDITS,
            "clear_thinking_20251015": ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value,
        }
    )

    def _drop_thinking(
        self,
        optional_params: MutableMapping[str, object],  # mutable-ok: edits optional params in place
        model: str,
        custom_llm_provider: str,
        max_tokens: int | None,
    ) -> None:
        super()._drop_thinking(optional_params, model, custom_llm_provider, max_tokens)
        if not self._adds_thinking_back_for_clear_thinking(optional_params, model, max_tokens):
            drop_clear_thinking_edits_for_model(optional_params, model, custom_llm_provider)

    def _adds_thinking_back_for_clear_thinking(
        self, optional_params: Mapping[str, object], model: str, max_tokens: int | None
    ) -> bool:
        trial_request: Final = {
            "context_management": optional_params.get("context_management"),
            "max_tokens": max_tokens,
        }
        return self._ensure_thinking_for_clear_thinking_context_management(trial_request, model, drop_params=False)

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
