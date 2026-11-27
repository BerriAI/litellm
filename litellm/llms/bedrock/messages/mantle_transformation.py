"""
Transformation for Bedrock Mantle (Claude Mythos Preview) - /messages endpoint

Rust sends /v1/messages calls on this route. Python keeps the config for provider lookups, the URL
and the request transform that count_tokens reads its thinking, tool_choice and output_config from.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.llms.bedrock.common_utils import build_mantle_messages_url
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams

_BODY_FIELDS_MANTLE_READS_FROM_HEADERS: Final = frozenset({"anthropic_version", "anthropic_beta"})
_ANTHROPIC_BETAS: Final = TypeAdapter(tuple[str, ...])
_MANTLE_REQUEST: Final = TypeAdapter(dict[str, object])


def _move_betas_into_header(request: Mapping[str, object], headers: dict[str, str]) -> None:
    betas: Final = _ANTHROPIC_BETAS.validate_python(request.get("anthropic_beta") or ())
    if betas:
        headers["anthropic-beta"] = ",".join(betas)  # rebind-ok: the handler signs and sends this same dict
        return
    headers.pop("anthropic-beta", None)


# TODO: `bedrock/mantle/` is slated for deprecation in favour of `bedrock_mantle/`. Folding it in must keep its defaults
# (us-west-2, no BEDROCK_MANTLE_API_KEY, build_mantle_messages_url, no claude gate) or ship with a deprecation notice.
class AmazonMantleMessagesConfig(AmazonAnthropicClaudeMessagesConfig):
    """
    Config for the bedrock-mantle /messages endpoint (Claude Mythos Preview).

    The mantle endpoint uses the Anthropic Messages API format and requires the
    model ID in the request body (unlike Bedrock Invoke which puts it in the URL).
    """

    BEDROCK_INVOKE_ALLOWED_TOP_LEVEL_FIELDS = (
        AmazonAnthropicClaudeMessagesConfig.BEDROCK_INVOKE_ALLOWED_TOP_LEVEL_FIELDS | {"service_tier"}
    )

    @property
    def beta_headers_provider(self) -> str:
        return "bedrock_mantle"

    def should_filter_anthropic_beta_headers(self) -> bool:
        return False

    def drops_dangerous_tool_use_beta_without_safeguards(self) -> bool:
        return False

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        region: Final = self._get_aws_region_name(optional_params=optional_params, model=model)
        return build_mantle_messages_url(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.get("aws_bedrock_runtime_endpoint"),
            region=region,
        )

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict[str, object]],
        anthropic_messages_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> dict[str, object]:
        model_id: Final = model.replace("mantle/", "", 1)
        request: Final = _MANTLE_REQUEST.validate_python(
            super().transform_anthropic_messages_request(
                model=model_id,
                messages=messages,
                anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            ),
        )
        _move_betas_into_header(request, headers)
        body: Final = MappingProxyType(
            {key: value for key, value in request.items() if key not in _BODY_FIELDS_MANTLE_READS_FROM_HEADERS}
        )
        streaming: Final = anthropic_messages_optional_request_params.get("stream") is True
        mantle_fields: Final = MappingProxyType(
            {key: value for key, value in (("model", model_id), ("stream", streaming)) if value}
        )
        return {
            **body,
            **mantle_fields,
        }
