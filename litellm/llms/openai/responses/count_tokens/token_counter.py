"""
OpenAI Token Counter implementation using the Responses API /input_tokens endpoint.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import openai
from pydantic import JsonValue
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.llms.anthropic.pass_through.convertible_blocks import deployment_drops_params, unconvertible_blocks_error
from litellm.llms.anthropic.pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.openai.common_utils import OpenAIError, get_openai_credentials
from litellm.llms.openai.responses.count_tokens.handler import (
    OpenAICountTokensHandler,
)
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
    ResponsesCountRequestParts,
    native_count_request,
)
from litellm.llms.openai.workload_identity import (
    OpenAIWorkloadIdentityConfig,
    get_workload_identity_bearer_token,
    openai_static_api_key,
    resolve_openai_workload_identity_config,
)
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

# Global handler instance - reuse across all token counting requests
openai_count_tokens_handler: Final = OpenAICountTokensHandler()


def _workload_identity_config(
    litellm_params: Mapping[str, object],
) -> OpenAIWorkloadIdentityConfig | OpenAIError | None:
    api_base: Final = litellm_params.get("api_base")
    try:
        return resolve_openai_workload_identity_config(
            api_key=None,
            api_base=api_base if isinstance(api_base, str) else None,
            litellm_params=litellm_params,
        )
    except OpenAIError as e:
        return e


@dataclass(frozen=True, slots=True)
class _CountCredential:
    api_base: str | None
    credential: str | OpenAIWorkloadIdentityConfig


def _count_credential(deployment: Mapping[str, object] | None, model_label: str) -> _CountCredential | OpenAIError:
    litellm_params: Final = as_str_mapping(None if deployment is None else deployment.get("litellm_params")) or {}
    api_base: Final = litellm_params.get("api_base")
    api_key: Final = litellm_params.get("api_key")
    credentials: Final = get_openai_credentials(
        api_base=api_base if isinstance(api_base, str) else None,
        api_key=api_key if isinstance(api_key, str) else None,
    )
    credential: Final = openai_static_api_key(credentials.api_key) or _workload_identity_config(litellm_params)
    if isinstance(credential, OpenAIError):
        return credential
    if credential is None:
        return OpenAIError(
            status_code=401,
            message=(
                f"No OpenAI API key is set for model {model_label}, so tokens can't be counted. Set api_key or "
                "workload identity on the deployment, or OPENAI_API_KEY in the proxy environment."
            ),
        )
    return _CountCredential(api_base=credentials.api_base, credential=credential)


def _count_error(model_to_use: str, request_model: str, message: str, status_code: int) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=0,
        request_model=request_model,
        model_used=model_to_use,
        tokenizer_type="openai_api",
        error=True,
        error_message=message,
        status_code=status_code,
    )


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter implementation for OpenAI provider using the Responses API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.OPENAI.value

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
                fallback_instructions: Final = OpenAITokenCounter._system_instructions(system)
                return OpenAICountTokensConfig.count_request_parts(
                    input_items, instructions if instructions is not None else fallback_instructions, tools
                )
            case _:
                assert_never(messages_format)

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, object]] | None,
        contents: list[dict[str, object]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, object]] | None = None,
        system: object = None,
        messages_format: TokenCountMessagesFormat = "openai",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        """
        Count tokens using OpenAI's Responses API /input_tokens endpoint.
        """
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
                tokenizer_type="openai_api",
                error=True,
                error_message=request_error,
                status_code=400,
                rejected_before_provider_call=True,
            )

        resolved: Final = _count_credential(deployment, request_model or model_to_use)
        if isinstance(resolved, OpenAIError):
            return _count_error(model_to_use, request_model, resolved.message, resolved.status_code)

        input_items, instructions, counted_tools, text = self._responses_request_parts(
            model_to_use=model_to_use,
            messages=messages,
            tools=tools,
            system=system,
            messages_format=messages_format,
            output_format=output_format,
        )

        # If no input items were produced (e.g., system-only messages), fall back to local counting
        if not input_items:
            return None

        return await self._counted(
            resolved,
            lambda api_key, api_base: openai_count_tokens_handler.handle_count_tokens_request(
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
        if isinstance(resolved, OpenAIError):
            return _count_error(model_to_use, request_model, resolved.message, resolved.status_code)
        body: Final = native_count_request(request, model_to_use, policy)
        return await self._counted(
            resolved,
            lambda api_key, api_base: openai_count_tokens_handler.post_count_request(body, api_key, api_base),
            model_to_use,
            request_model,
        )

    @staticmethod
    async def _counted(
        resolved: _CountCredential,
        send: Callable[[str, str | None], Awaitable[Mapping[str, object]]],
        model_to_use: str,
        request_model: str,
    ) -> TokenCountResponse | None:
        try:
            api_key: Final = (
                resolved.credential
                if isinstance(resolved.credential, str)
                else await asyncify(get_workload_identity_bearer_token)(resolved.credential)
            )
            result: Final = await send(api_key, resolved.api_base)

            if result is not None:
                return TokenCountResponse.model_validate(
                    {
                        "total_tokens": result.get("input_tokens", 0),
                        "request_model": request_model,
                        "model_used": model_to_use,
                        "tokenizer_type": "openai_api",
                        "original_response": result,
                    }
                )
        except OpenAIError as e:
            verbose_logger.warning("OpenAI CountTokens API error: status=%s, message=%s", e.status_code, e.message)
            return _count_error(model_to_use, request_model, e.message, e.status_code)
        except openai.AuthenticationError as e:
            verbose_logger.warning("OpenAI workload identity token exchange failed: %s", e.message)
            return _count_error(model_to_use, request_model, e.message, 401)
        except Exception as e:
            verbose_logger.warning("Error calling OpenAI CountTokens API: %s", e)
            return _count_error(model_to_use, request_model, str(e), 500)

        return None
