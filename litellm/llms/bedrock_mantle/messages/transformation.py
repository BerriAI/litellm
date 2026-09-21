from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

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
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES
from litellm.types.router import GenericLiteLLMParams

_BASE_SUFFIXES_TO_STRIP: Final = (
    MANTLE_MESSAGES_PATH,
    "/v1/messages",
    "/messages",
    "/anthropic/v1",
    "/openai/v1",
    "/v1",
)
_BODY_FIELDS_MANTLE_READS_FROM_HEADERS: Final = frozenset({"anthropic_version", "anthropic_beta"})
_ANTHROPIC_BETAS: Final = TypeAdapter(tuple[str, ...])
_MANTLE_REQUEST: Final = TypeAdapter(dict[str, object])


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
        return {  # mutable-ok: the base class contract returns a dict the handler signs into in place
            **merged_headers,
            "anthropic-version": DEFAULT_ANTHROPIC_API_VERSION,
        }, resolved_api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request: Final = _MANTLE_REQUEST.validate_python(
            super().transform_anthropic_messages_request(
                model=model,
                messages=messages,
                anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            ),
        )
        betas: Final = request.get("anthropic_beta")
        if betas is not None:
            header_betas: Final = ",".join(_ANTHROPIC_BETAS.validate_python(betas))
            headers["anthropic-beta"] = header_betas  # rebind-ok: the handler signs and sends this same dict
        return {  # mutable-ok: the base class contract returns the dict the handler serializes as the body
            key: value for key, value in request.items() if key not in _BODY_FIELDS_MANTLE_READS_FROM_HEADERS
        }
