from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

import httpx
from pydantic import JsonValue
from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping, decoded_json_object
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import parts_of
from litellm.llms.anthropic.messages.context_management import AnthropicContextManagementError
from litellm.llms.anthropic.messages.convertible_blocks import deployment_drops_params, unconvertible_blocks_error
from litellm.llms.anthropic.messages.responses_adapters.handler import count_messages_as_sent
from litellm.llms.anthropic.messages.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.llms.mistral.common_utils import MistralError, get_mistral_chat_credentials
from litellm.llms.mistral.count_tokens.handler import (
    MistralCountTokensHandler,
)
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
    ResponsesCountRequestParts,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse
from litellm.utils import catalog_with_runtime_registrations

mistral_count_tokens_handler: Final = MistralCountTokensHandler()
_MISTRAL_CHAT_CONFIG: Final = MistralConfig()
_CHAT_TO_RESPONSES: Final = LiteLLMResponsesTransformationHandler()
_MODES_MISTRAL_COUNTS: Final = frozenset({"chat", "responses"})
_REASONING_PROMPT_PARAMS: Final = ("reasoning_effort", "thinking")


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
    azure_key: Final = get_secret_str("MISTRAL_AZURE_API_KEY")
    if _is_azure_endpoint(api_base) or (bool(azure_key) and api_key == azure_key):
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


def _counts_on_mistral(model: str) -> bool:
    cost_entry: Final = catalog_with_runtime_registrations().get(
        model if model.startswith("mistral/") else f"mistral/{model}"
    )
    mode: Final = None if cost_entry is None else cost_entry.get("mode")
    return mode is None or mode in _MODES_MISTRAL_COUNTS


def _deployment_timeout(deployment: Mapping[str, object] | None) -> float | httpx.Timeout | None:
    litellm_params: Final = as_str_mapping(None if deployment is None else deployment.get("litellm_params")) or {}
    match litellm_params.get("timeout"):
        case bool():
            return None
        case httpx.Timeout() as timeout:
            return timeout
        case int(seconds) | float(seconds):
            return float(seconds)
        case _:
            return None


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


def _file_id(block: object) -> str | None:
    match block:
        case {"type": "file", "file": {"file_id": str(file_id)}} if file_id:
            return file_id
        case {"type": "file", "file_id": str(file_id)} if file_id:
            return file_id
        case _:
            return None


def _file_id_parts(message: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    blocks: Final = parts_of(message.get("content")) if message.get("role") == "user" else ()
    file_ids: Final = tuple(file_id for file_id in map(_file_id, blocks) if file_id is not None)
    return tuple({"type": "input_file", "file_id": file_id} for file_id in file_ids)


def _chat_drops_turn(message: Mapping[str, object]) -> bool:
    return (
        message.get("role") == "assistant"
        and message.get("content") in (None, "")
        and message.get("tool_calls") is None
    )


def _replayed_reasoning_items(message: Mapping[str, object], keeps_reasoning: bool) -> tuple[Mapping[str, object], ...]:
    if not keeps_reasoning or message.get("role") != "assistant":
        return ()
    if not message.get("content") and not message.get("tool_calls"):
        return ()
    reasoning_text: Final = MistralConfig.replayed_reasoning_text(message)
    if reasoning_text is None:
        return ()
    return ({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning_text}]},)


def _input_content_parts(content: object) -> tuple[object, ...]:
    match content:
        case str(text) if text:
            return ({"type": "input_text", "text": text},)
        case _:
            return parts_of(content)


def _part_text(part: object) -> str:
    match part:
        case {"text": str(text)}:
            return text
        case _:
            return ""


def _system_text(content: object) -> str:
    return content if isinstance(content, str) else "".join(map(_part_text, parts_of(content)))


def _is_system_turn(message: Mapping[str, object]) -> bool:
    return message.get("role") in ("system", "developer")


def _chat_message_items(message: Mapping[str, object], keeps_reasoning: bool) -> tuple[Mapping[str, object], ...]:
    if _is_system_turn(message):
        return ({"role": "system", "content": _system_text(message.get("content"))},)
    items, _ = OpenAICountTokensConfig.messages_to_responses_input((message,))
    file_parts: Final = _file_id_parts(message)
    sent_items: Final = (
        tuple({**item, "content": [*_input_content_parts(item.get("content")), *file_parts]} for item in items)
        if file_parts
        else tuple(items)
    )
    return (*_replayed_reasoning_items(message, keeps_reasoning), *sent_items)


def _counted_tool_choice(tool_choice: object) -> JsonValue:
    match MistralConfig.map_tool_choice(tool_choice):
        case "any":
            return "required"
        case {"function": {"name": str(name)}}:
            return {"type": "function", "name": name}
        case str(choice):
            return choice
        case _:
            return None


def _with_chat_reasoning_prompt(
    model: str, messages: Sequence[Mapping[str, object]], chat_params: Mapping[str, JsonValue]
) -> Sequence[Mapping[str, object]]:
    if "magistral" not in model.lower() or all(chat_params.get(name) is None for name in _REASONING_PROMPT_PARAMS):
        return messages
    return MistralConfig.with_reasoning_system_prompt(messages)


class MistralTokenCounter(BaseTokenCounter):
    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.MISTRAL.value

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
                keeps_reasoning: Final = _MISTRAL_CHAT_CONFIG.accepts_reasoning(model_to_use)
                input_items: Final = tuple(
                    chain.from_iterable(
                        _chat_message_items(message, keeps_reasoning)
                        for message in messages
                        if not _chat_drops_turn(message)
                    )
                )
                return OpenAICountTokensConfig.count_request_parts(
                    input_items,
                    None
                    if any(map(_is_system_turn, messages))
                    else LiteLLMAnthropicToResponsesAPIAdapter.system_instructions(system),
                    tools,
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
        if not messages or not _counts_on_mistral(model_to_use):
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

        sent_messages: Final = await count_messages_as_sent(
            model=model_to_use,
            messages=messages,
            tools=tools,
            system=system,
            messages_format=messages_format,
            context_management=context_management,
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
            messages=sent_messages
            if messages_format == "anthropic"
            else _with_chat_reasoning_prompt(model_to_use, sent_messages, chat_params),
            tools=tools,
            system=system,
            messages_format=messages_format,
            output_format=output_format,
        )

        if not input_items:
            return None
        counted_text: Final = (
            text
            if text is not None
            else _CHAT_TO_RESPONSES.transform_response_format_to_text_format(chat_params.get("response_format"))
        )
        timeout: Final = _deployment_timeout(deployment)
        tool_choice: Final = _counted_tool_choice(chat_params.get("tool_choice")) if counted_tools else None

        return await self._counted(
            resolved,
            lambda api_key, api_base: mistral_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                input=input_items,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                tools=counted_tools,
                instructions=instructions,
                text=counted_text,
                tool_choice=tool_choice,
            ),
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
            input_tokens: Final = None if result is None else decoded_json_object(result).get("input_tokens")
            if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
                verbose_logger.warning("Mistral count response had no input_tokens, falling back to local: %s", result)
                return None
            return TokenCountResponse.model_validate(
                {
                    "total_tokens": input_tokens,
                    "request_model": request_model,
                    "model_used": model_to_use,
                    "tokenizer_type": "mistral_api",
                    "original_response": result,
                }
            )
