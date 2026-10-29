import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from itertools import groupby
from types import MappingProxyType
from typing import ClassVar, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.constants import XAI_API_BASE
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    MaskedHTTPStatusError,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # its params argument is a bare dict
)
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

TOKENIZER_TYPE: Final = "xai_api"
_SEPARATOR: Final = "<|separator|>\n\n"
_ROLE_LABELS: Final = MappingProxyType({"system": "System", "developer": "System", "assistant": "Assistant"})


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)


class _TextPart(_Strict):
    type: Literal["text"]
    text: str


class _Function(_Strict):
    name: str
    arguments: str


class _ToolCall(_Strict):
    function: _Function


class _Message(_Strict):
    role: str
    content: str | tuple[_TextPart, ...] | None = None
    tool_calls: tuple[_ToolCall, ...] | None = None

    @property
    def label(self) -> str:
        return _ROLE_LABELS.get(self.role, "Human")

    @property
    def text(self) -> str:
        content: Final = (
            self.content if isinstance(self.content, str) else "\n".join(part.text for part in self.content or ())
        )
        calls: Final = (f"{call.function.name}({call.function.arguments})" for call in self.tool_calls or ())
        return "\n".join(filter(None, (content.strip(), *calls)))


class _TranslatedRequest(_Strict):
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...] | None = None


class _TokenizeTextReply(_Strict):
    token_ids: tuple[JsonValue, ...]


_MESSAGES: Final = TypeAdapter(tuple[_Message, ...], config=ConfigDict(hide_input_in_errors=True))
_LITELLM_PARAMS: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


def _turn(label: str, messages: Iterable[_Message]) -> str:
    return f"{label}: " + "\n".join(message.text for message in messages) + _SEPARATOR


def rendered_grok_prompt(
    messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]] | None
) -> str | None:
    try:
        parsed: Final = _MESSAGES.validate_python(messages)
    except ValidationError:
        return None
    tool_turn: Final = (f"System: {json.dumps(list(tools), ensure_ascii=False)}{_SEPARATOR}",) if tools else ()
    turns: Final = (_turn(label, grouped) for label, grouped in groupby(parsed, key=lambda message: message.label))
    return "".join((*tool_turn, *turns))


def _openai_shape(
    model: str,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    system: object,
    messages_format: TokenCountMessagesFormat,
) -> tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]] | None] | None:
    if messages_format == "openai":
        system_turn: Final[Mapping[str, object]] = {"role": "system", "content": system}
        leading: Final = (system_turn,) if isinstance(system, (str, list)) and system else ()
        return (*leading, *messages), tools
    body: Final = {
        "model": model,
        "max_tokens": 1,
        "messages": copy.deepcopy(list(messages)),
        **({"system": copy.deepcopy(system)} if system else {}),
        **({"tools": copy.deepcopy(list(tools))} if tools else {}),
    }
    try:
        translated, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            body,  # pyright: ignore[reportArgumentType]  # raw /v1/messages fields
            custom_llm_provider=LlmProviders.XAI.value,
        )
    except Exception:  # noqa: BLE001  # a request the adapter cannot translate falls back to the estimate
        return None
    shaped: Final = _TranslatedRequest.model_validate(translated)
    return shaped.messages, shaped.tools


def _is_xai_deployment(litellm_params: Mapping[str, object]) -> bool:
    model: Final = litellm_params.get("model")
    provider_hint: Final = litellm_params.get("custom_llm_provider")
    if not isinstance(model, str):
        return False
    try:
        _, provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=provider_hint if isinstance(provider_hint, str) else None
        )
    except Exception:  # noqa: BLE001  # get_llm_provider raises bare exceptions for unknown models
        return False
    return provider == LlmProviders.XAI.value


def _non_blank(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


class XAITokenCounter(BaseTokenCounter):
    includes_framing: ClassVar[bool] = False

    def __init__(self, client: AsyncHTTPHandler | None = None) -> None:
        self._client: Final = client

    def should_use_token_counting_api(self, custom_llm_provider: str | None = None) -> bool:
        return True

    def _error(self, request_model: str, model: str, status_code: int, message: str) -> TokenCountResponse:
        verbose_logger.warning("xAI tokenize-text error: status=%s, message=%s", status_code, message)
        return TokenCountResponse(
            total_tokens=0,
            request_model=request_model,
            model_used=model,
            tokenizer_type=TOKENIZER_TYPE,
            error=True,
            error_message=message,
            status_code=status_code,
        )

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Sequence[Mapping[str, object]] | None,
        contents: Sequence[Mapping[str, object]] | None,
        deployment: Mapping[str, object] | None = None,
        request_model: str = "",
        tools: Sequence[Mapping[str, object]] | None = None,
        system: object | None = None,
        messages_format: TokenCountMessagesFormat = "openai",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        if not messages or contents:
            return None
        model: Final = model_to_use.removeprefix(f"{LlmProviders.XAI.value}/")
        shaped: Final = _openai_shape(model, messages, tools, system, messages_format)
        text: Final = None if shaped is None else rendered_grok_prompt(*shaped)
        if not text:
            return None
        litellm_params: Final = _LITELLM_PARAMS.validate_python((deployment or {}).get("litellm_params") or {})
        xai_deployment: Final = _is_xai_deployment(litellm_params)
        credentials: Final[Mapping[str, object]] = litellm_params if xai_deployment else {}
        api_key: Final = XAIModelInfo.get_api_key(_non_blank(credentials.get("api_key")))
        if api_key is None and xai_deployment:
            return None
        if api_key is None:
            return self._error(
                request_model,
                model,
                401,
                f"No xAI API key is set for model {request_model or model}, so tokens can't be counted. "
                "Set XAI_API_KEY in the proxy environment.",
            )
        api_base: Final = _non_blank(credentials.get("api_base")) or get_secret_str("XAI_API_BASE") or XAI_API_BASE
        client: Final = self._client or get_async_httpx_client(llm_provider=LlmProviders.XAI)
        try:
            response: Final = await client.post(  # pyright: ignore[reportUnknownMemberType]  # post takes bare dicts
                url=join_api_base(api_base, "tokenize-text"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"text": text, "model": model},
            )
            reply: Final = _TokenizeTextReply.model_validate_json(response.content)
        except MaskedHTTPStatusError as e:
            status_code: Final = e.response.status_code
            return self._error(request_model, model, status_code, e.text or f"xAI tokenize-text answered {status_code}")
        except (httpx.RequestError, litellm.Timeout, ValueError) as e:
            return self._error(request_model, model, 502, f"xAI tokenize-text failed: {type(e).__name__}")
        return TokenCountResponse(
            total_tokens=len(reply.token_ids),
            request_model=request_model,
            model_used=model,
            tokenizer_type=TOKENIZER_TYPE,
        )
