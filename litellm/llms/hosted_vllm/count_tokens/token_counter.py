"""
hosted_vllm Token Counter implementation using the vLLM Anthropic CountTokens API.

vLLM renders the model's own chat template, so its count matches what generation
consumes for the same prompt, including image blocks the local estimator only approximates.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountFields,
    ChatCountTranslationError,
    anthropic_count_fields,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

hosted_vllm_count_tokens_handler: Final = AnthropicCountTokensHandler()

_ABSENT_ENDPOINT_STATUS_CODES: Final = frozenset({404, 405})
_INPUT_TOKENS: Final = TypeAdapter(int)
_DECODED_JSON: Final = TypeAdapter(object)


def _is_absent_endpoint_error(status_code: int, message: str) -> bool:
    """A 404/405 whose body is not an Anthropic error envelope means the server has no
    count_tokens route, so the count falls back to the local estimate instead of failing.
    An Anthropic-shaped 404 is the backend rejecting the model and stays an error."""
    if status_code not in _ABSENT_ENDPOINT_STATUS_CODES:
        return False
    try:
        parsed: Final = _DECODED_JSON.validate_json(message)
    except ValidationError:
        return True
    body: Final = as_str_mapping(parsed)
    return body is None or body.get("type") != "error"


def _input_tokens(payload: Mapping[str, JsonValue]) -> int:
    try:
        return _INPUT_TOKENS.validate_python(payload.get("input_tokens", 0))
    except ValidationError:
        return 0


def _auth_header(api_key: object) -> Mapping[str, str]:
    """vLLM commonly runs keyless, so an unset key sends no auth header at all."""
    return {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key else {}


class HostedVLLMTokenCounter(BaseTokenCounter):
    """Count tokens by forwarding to a vLLM server's Anthropic count_tokens endpoint."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.HOSTED_VLLM.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Sequence[Mapping[str, JsonValue]] | None,
        contents: Sequence[Mapping[str, object]] | None,
        deployment: Mapping[str, object] | None = None,
        request_model: str = "",
        tools: Sequence[Mapping[str, JsonValue]] | None = None,
        system: JsonValue = None,
        messages_format: TokenCountMessagesFormat = "anthropic",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        if not messages:
            return None

        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        translated: Final = await anthropic_count_fields(
            model_to_use, messages, system, tools, messages_format, AnthropicConfig(), chat_params
        )
        if isinstance(translated, ChatCountTranslationError):
            return translated.response(request_model, model_to_use, "hosted_vllm_api")
        counted: Final = (
            translated
            if isinstance(translated, AnthropicCountFields)
            else AnthropicCountFields(
                messages=tuple(messages),
                system=system,
                tools=None if tools is None else tuple(tools),
                options=MappingProxyType({}),
            )
        )

        deployment_params: Final = as_str_mapping(None if deployment is None else deployment.get("litellm_params"))
        api_base: Final = None if deployment_params is None else deployment_params.get("api_base")
        if not isinstance(api_base, str) or not api_base:
            return None
        api_key: Final = None if deployment_params is None else deployment_params.get("api_key")

        try:
            result: Final = await hosted_vllm_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                messages=[dict(message) for message in counted.messages],
                auth_header=_auth_header(api_key),
                api_base=api_base,
                tools=None if counted.tools is None else [dict(tool) for tool in counted.tools],
                system=counted.system,
                optional_params=counted.options,
            )
        except AnthropicError as e:
            if _is_absent_endpoint_error(e.status_code, str(e.message)):
                verbose_logger.debug(
                    "hosted_vllm backend has no count_tokens route (status=%s), using the local estimate",
                    e.status_code,
                )
                return None
            verbose_logger.warning("hosted_vllm CountTokens error: status=%s, message=%s", e.status_code, e.message)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="hosted_vllm_api",
                error=True,
                error_message=str(e.message),
                status_code=e.status_code,
            )

        return TokenCountResponse(
            total_tokens=_input_tokens(result),
            request_model=request_model,
            model_used=model_to_use,
            tokenizer_type="hosted_vllm_api",
            original_response=dict(result),
        )
