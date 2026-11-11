from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

import httpx
from pydantic import JsonValue
from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, decoded_json_object
from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.llms.anthropic.messages.context_management import AnthropicContextManagementError
from litellm.llms.anthropic.messages.convertible_blocks import deployment_drops_params, unconvertible_blocks_error
from litellm.llms.anthropic.messages.responses_adapters.handler import count_view_messages
from litellm.llms.anthropic.messages.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.mistral.common_utils import MistralError, get_mistral_chat_credentials
from litellm.llms.mistral.count_tokens.handler import (
    MistralCountTokensHandler,
)
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
    ResponsesCountRequestParts,
    native_count_request,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

mistral_count_tokens_handler: Final = MistralCountTokensHandler()


@dataclass(frozen=True, slots=True)
class _MistralCountCredential:
    api_key: str
    api_base: str | None


def _is_azure_endpoint(api_base: str) -> bool:
    return "azure" in (urlparse(api_base).hostname or "").lower()


def _count_credential(
    deployment: Mapping[str, object] | None, model_label: str
) -> _MistralCountCredential | MistralError | None:
    litellm_params: Final = as_str_mapping(None if deployment is None else deployment.get("litellm_params")) or {}
    if litellm_params.get("custom_llm_provider") in ("azure", "azure_ai"):
        return None
    raw_api_base: Final = litellm_params.get("api_base")
    raw_api_key: Final = litellm_params.get("api_key")
    deployment_key: Final = raw_api_key if isinstance(raw_api_key, str) and raw_api_key else None
    api_base, api_key = get_mistral_chat_credentials(
        api_base=raw_api_base if isinstance(raw_api_base, str) else None, api_key=deployment_key
    )
    if _is_azure_endpoint(api_base) or (deployment_key is None and get_secret_str("MISTRAL_AZURE_API_KEY")):
        verbose_logger.debug(
            "Mistral native count tokens skipped for Azure deployment/credential, falling back to local: base=%s",
            api_base,
        )
        return None

    resolved_key: Final = api_key or litellm.api_key
    if not resolved_key:
        return MistralError(
            status_code=401,
            message=(
                f"No Mistral API key is set for model {model_label}, so tokens can't be counted. Set api_key on "
                "the deployment or MISTRAL_API_KEY in the environment."
            ),
            headers=httpx.Headers(),
        )
    return _MistralCountCredential(api_key=resolved_key, api_base=api_base)


def _count_error(model_to_use: str, request_model: str, message: str, status_code: int) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=0,
        request_model=request_model,
        model_used=model_to_use,
        tokenizer_type="mistral_api",
        error=True,
        error_message=message,
        status_code=status_code,
    )


async def _sent_messages(
    model: str,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    system: object,
    messages_format: TokenCountMessagesFormat,
    context_management: CountContextManagement | None,
) -> Sequence[Mapping[str, object]] | AnthropicContextManagementError:
    match messages_format:
        case "openai":
            return messages
        case "anthropic":
            try:
                return await count_view_messages(
                    model=model,
                    messages=messages,
                    tools=tools,
                    system=system,
                    context_management=context_management,
                )
            except AnthropicContextManagementError as error:
                return error
        case _:
            assert_never(messages_format)


class MistralTokenCounter(BaseTokenCounter):
    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.MISTRAL.value

    @staticmethod
    def _system_instructions(system: object) -> str | None:
        if system is None or isinstance(system, str):
            return system
        return LiteLLMAnthropicToResponsesAPIAdapter.system_blocks_to_instructions(system) or None

    @staticmethod
    def _responses_request_parts(
        model_to_use: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None,
        system: object,
        messages_format: TokenCountMessagesFormat,
        output_format: JsonValue,
    ) -> ResponsesCountRequestParts:
        match messages_format:
            case "anthropic":
                return OpenAICountTokensConfig.anthropic_messages_to_responses_input(
                    model=model_to_use, messages=messages, system=system, tools=tools, output_format=output_format
                )
            case "openai":
                input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
                    [dict(message) for message in messages]
                )
                fallback_instructions: Final = MistralTokenCounter._system_instructions(system)
                return OpenAICountTokensConfig.count_request_parts(
                    input_items, instructions if instructions is not None else fallback_instructions, tools
                )
            case _:
                assert_never(messages_format)

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, object]] | None,  # mutable-ok: matches BaseTokenCounter signature
        contents: list[dict[str, object]] | None,  # mutable-ok: matches BaseTokenCounter signature
        deployment: dict[str, object] | None = None,  # mutable-ok: matches BaseTokenCounter signature
        request_model: str = "",
        tools: list[dict[str, object]] | None = None,  # mutable-ok: matches BaseTokenCounter signature
        system: object = None,
        messages_format: TokenCountMessagesFormat = "openai",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        if not messages:
            return None

        malformed_error: Final = (
            OpenAICountTokensConfig.malformed_anthropic_messages_error(
                messages, drop_params=deployment_drops_params(deployment)
            )
            if messages_format == "anthropic"
            else None
        )
        request_error: Final = malformed_error or unconvertible_blocks_error(messages, messages_format)
        if request_error is not None:
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="mistral_api",
                error=True,
                error_message=request_error,
                status_code=400,
                rejected_before_provider_call=True,
            )

        sent_messages: Final = await _sent_messages(
            model_to_use, messages, tools, system, messages_format, context_management
        )
        if isinstance(sent_messages, AnthropicContextManagementError):
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="mistral_api",
                error=True,
                error_message=sent_messages.message,
                status_code=sent_messages.status_code,
                rejected_before_provider_call=True,
            )

        resolved: Final = _count_credential(deployment, request_model or model_to_use)
        if resolved is None:
            return None
        if isinstance(resolved, MistralError):
            return _count_error(model_to_use, request_model, resolved.message, resolved.status_code)

        input_items, instructions, counted_tools, text = self._responses_request_parts(
            model_to_use=model_to_use,
            messages=sent_messages,
            tools=tools,
            system=system,
            messages_format=messages_format,
            output_format=output_format,
        )

        if not input_items:
            return None

        return await self._counted(
            resolved,
            lambda api_key, api_base: mistral_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                input=input_items,
                api_key=api_key,
                api_base=api_base,
                tools=counted_tools,
                instructions=instructions,
                text=text,
            ),
            model_to_use,
            request_model,
        )

    async def count_responses_request(
        self,
        model_to_use: str,
        request: Mapping[str, object],
        policy: DropPolicy,
        deployment: Mapping[str, object] | None,
        request_model: str,
    ) -> TokenCountResponse | None:
        resolved: Final = _count_credential(deployment, request_model or model_to_use)
        if resolved is None:
            return None
        if isinstance(resolved, MistralError):
            return _count_error(model_to_use, request_model, resolved.message, resolved.status_code)
        body: Final = native_count_request(request, model_to_use, policy)
        return await self._counted(
            resolved,
            lambda api_key, api_base: mistral_count_tokens_handler.post_count_request(body, api_key, api_base),
            model_to_use,
            request_model,
        )

    @staticmethod
    async def _counted(
        resolved: _MistralCountCredential,
        send: Callable[[str, str | None], Awaitable[object]],
        model_to_use: str,
        request_model: str,
    ) -> TokenCountResponse | None:
        try:
            result: Final = await send(resolved.api_key, resolved.api_base)
        except (httpx.RequestError, litellm.Timeout, litellm.APIConnectionError, OSError) as e:
            verbose_logger.warning("Mistral count tokens connection error / timeout, falling back to local: %s", e)
            return None
        except MistralError as e:
            verbose_logger.warning("Mistral CountTokens API error: status=%s, message=%s", e.status_code, e.message)
            return _count_error(model_to_use, request_model, e.message, e.status_code)
        except (httpx.HTTPError, ValueError) as e:
            verbose_logger.warning("Error calling Mistral CountTokens API: %s", e)
            return _count_error(model_to_use, request_model, str(e), 500)
        else:
            if result is not None:
                return TokenCountResponse.model_validate(
                    {
                        "total_tokens": decoded_json_object(result).get("input_tokens", 0),
                        "request_model": request_model,
                        "model_used": model_to_use,
                        "tokenizer_type": "mistral_api",
                        "original_response": result,
                    }
                )
            return None
