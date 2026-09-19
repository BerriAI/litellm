from collections.abc import Mapping
from typing import Final

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    DEFAULT_ANTHROPIC_API_VERSION,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import MANTLE_MESSAGES_PATH
from litellm.llms.bedrock.messages.mantle_transformation import AmazonMantleMessagesConfig
from litellm.llms.bedrock_mantle.common_utils import (
    MANTLE_HOST_RE,
    BedrockMantleAuthMixin,
    resolve_mantle_region,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

_BASE_SUFFIXES_TO_STRIP: Final = (
    MANTLE_MESSAGES_PATH,
    "/v1/messages",
    "/messages",
    "/anthropic/v1",
    "/openai/v1",
    "/v1",
)


def build_mantle_native_messages_url(api_base: str | None, litellm_params: Mapping[str, object]) -> str:
    region: Final = resolve_mantle_region({**litellm_params, "api_base": api_base})
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
    def __init__(self, aws_signer: BaseAWSLLM | None = None) -> None:
        AmazonMantleMessagesConfig.__init__(self)
        self._aws_signer = aws_signer or self

    @property
    def custom_llm_provider(self) -> str | None:
        return "bedrock_mantle"

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

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: list[dict],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        merged_headers, resolved_api_base = super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        if any(name.lower() == "anthropic-version" for name in merged_headers):
            return merged_headers, resolved_api_base
        return {**merged_headers, "anthropic-version": DEFAULT_ANTHROPIC_API_VERSION}, resolved_api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request: Final = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if "anthropic_version" in anthropic_messages_optional_request_params:
            return request
        return {key: value for key, value in request.items() if key != "anthropic_version"}
