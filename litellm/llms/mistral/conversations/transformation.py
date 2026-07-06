"""
Transformation for Mistral's Conversations API (``POST /v1/conversations``).

Mistral's built-in ``web_search`` / ``web_search_premium`` connectors are only
available through the Conversations API; they are not supported on
``/v1/chat/completions``. This config lets a normal chat-completion request that
asks for web search (via ``tools=[{"type": "web_search"}]`` or
``web_search_options``) route to the Conversations endpoint and maps the
response back to an OpenAI-shaped ``ModelResponse`` with ``url_citation``
annotations for the web sources.

Docs - https://docs.mistral.ai/agents/connectors/websearch/
"""

from typing import Optional

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.llms.mistral.common_utils import OBJ_LIST, STR_OBJ_DICT, WEB_SEARCH_TOOL_TYPES
from litellm.types.llms.mistral import (
    MistralConversationContentChunk,
    MistralConversationOutput,
    MistralConversationsResponse,
    MistralConversationUsage,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
)
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

_COMPLETION_ARG_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "response_format",
    "random_seed",
)


def _block_text(block: object) -> str:
    if not isinstance(block, dict):
        return ""
    typed = STR_OBJ_DICT.validate_python(block)
    text = typed.get("text")
    if typed.get("type") == "text" and isinstance(text, str):
        return text
    return ""


def _content_to_str(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_text(block) for block in OBJ_LIST.validate_python(content))
    return "" if content is None else str(content)


def _function_call_entry(call: dict[str, object]) -> dict[str, object]:
    fn = STR_OBJ_DICT.validate_python(call["function"]) if isinstance(call.get("function"), dict) else {}
    return {
        "type": "function.call",
        "tool_call_id": str(call.get("id") or ""),
        "name": str(fn.get("name") or ""),
        "arguments": str(fn.get("arguments") or ""),
    }


def _to_annotation(chunk: MistralConversationContentChunk) -> ChatCompletionAnnotation:
    url_citation: ChatCompletionAnnotationURLCitation = (
        {"url": chunk.url or "", "title": chunk.title} if chunk.title else {"url": chunk.url or ""}
    )
    return {"type": "url_citation", "url_citation": url_citation}


class MistralConversationsConfig(MistralConfig):
    @property
    def reserved_request_body_keys(self) -> frozenset[str]:
        """Sanitized Conversations fields extra_body may not clobber: ``tools`` is
        allowlist-checked, ``store`` is pinned to False, and ``inputs`` / ``model``
        are built from the authenticated request."""
        return frozenset({"tools", "store", "inputs", "model"})

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        return bool(stream)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = (api_base or "https://api.mistral.ai/v1").rstrip("/")
        return f"{base}/conversations"

    @staticmethod
    def _build_tools(optional_params: dict[str, object]) -> list[dict[str, object]]:
        raw_tools = optional_params.get("tools")
        candidate_tools = OBJ_LIST.validate_python(raw_tools) if isinstance(raw_tools, list) else []
        dict_tools = [STR_OBJ_DICT.validate_python(tool) for tool in candidate_tools if isinstance(tool, dict)]

        passthrough = tuple(tool for tool in dict_tools if tool.get("type") == "function")
        web_search_options = optional_params.get("web_search_options")
        wants_web_search = web_search_options is not None or any(
            tool.get("type") in WEB_SEARCH_TOOL_TYPES for tool in dict_tools
        )
        premium = any(tool.get("type") == "web_search_premium" for tool in dict_tools) or (
            isinstance(web_search_options, dict)
            and STR_OBJ_DICT.validate_python(web_search_options).get("premium") is True
        )

        web_search_tool: tuple[dict[str, object], ...] = (
            ({"type": "web_search_premium" if premium else "web_search"},) if wants_web_search else ()
        )
        return list(web_search_tool + passthrough)

    @staticmethod
    def _build_completion_args(optional_params: dict[str, object]) -> dict[str, object]:
        return {key: optional_params[key] for key in _COMPLETION_ARG_KEYS if optional_params.get(key) is not None}

    @staticmethod
    def _input_entries_for_message(message: AllMessageValues) -> tuple[dict[str, object], ...]:
        """Map one OpenAI message to its Conversations ``inputs`` entries.

        A ``tool`` message becomes a ``function.result`` entry; an assistant
        message's ``tool_calls`` each become a ``function.call`` entry (preserving
        the id/name/arguments binding); text content becomes a plain message
        entry. Preserving this history keeps a web-search turn mid-conversation
        from silently dropping prior tool calls the way a role+content flatten would.
        """
        typed = STR_OBJ_DICT.validate_python(message)
        role = str(typed.get("role") or "")
        if role == "tool":
            return (
                {
                    "type": "function.result",
                    "tool_call_id": str(typed.get("tool_call_id") or ""),
                    "result": _content_to_str(typed.get("content")),
                },
            )
        raw_calls = typed.get("tool_calls") if role == "assistant" else None
        calls = OBJ_LIST.validate_python(raw_calls) if isinstance(raw_calls, list) else []
        content = _content_to_str(typed.get("content"))
        message_entry: tuple[dict[str, object], ...] = (
            ({"role": role, "content": content},) if content or not calls else ()
        )
        call_entries = tuple(
            _function_call_entry(STR_OBJ_DICT.validate_python(call)) for call in calls if isinstance(call, dict)
        )
        return message_entry + call_entries

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict[str, object]:
        params = STR_OBJ_DICT.validate_python(optional_params)
        instructions = "\n\n".join(
            _content_to_str(message.get("content"))
            for message in messages
            if message.get("role") == "system" and _content_to_str(message.get("content"))
        )
        inputs = [
            entry
            for message in messages
            if str(message.get("role")) != "system"
            for entry in self._input_entries_for_message(message)
        ]
        completion_args = self._build_completion_args(params)
        return {
            "model": model,
            "inputs": inputs,
            "tools": self._build_tools(params),
            "store": False,
            **({"instructions": instructions} if instructions else {}),
            **({"completion_args": completion_args} if completion_args else {}),
        }

    @staticmethod
    def _extract_message(
        outputs: list[MistralConversationOutput],
    ) -> tuple[str, list[ChatCompletionAnnotation]]:
        message_output = next((output for output in outputs if output.type == "message.output"), None)
        if message_output is None:
            return "", []
        content = message_output.content
        if isinstance(content, str):
            return content, []
        if not isinstance(content, list):
            return "", []
        text = "".join(chunk.text for chunk in content if chunk.type == "text" and chunk.text)
        annotations = [_to_annotation(chunk) for chunk in content if chunk.type == "tool_reference" and chunk.url]
        return text, annotations

    @staticmethod
    def _count_web_searches(
        usage: Optional[MistralConversationUsage],
        outputs: list[MistralConversationOutput],
    ) -> tuple[int, int]:
        """Return ``(standard, premium)`` web search call counts.

        Prefers the authoritative per-connector billed counts in
        ``usage.connectors``. Falls back to counting ``tool.execution`` entries by
        name; web search is the only built-in connector LiteLLM sends, so an
        unnamed execution is treated as a standard web search and other named
        connectors are excluded.
        """
        connectors = usage.connectors if usage else None
        if connectors:
            return int(connectors.get("web_search", 0) or 0), int(connectors.get("web_search_premium", 0) or 0)
        executions = [output for output in outputs if output.type == "tool.execution"]
        premium = sum(1 for output in executions if output.name == "web_search_premium")
        standard = sum(1 for output in executions if output.name in ("web_search", None))
        return standard, premium

    @staticmethod
    def _build_usage(
        usage: Optional[MistralConversationUsage],
        web_search_requests: int,
        web_search_premium_requests: int,
    ) -> Usage:
        prompt_tokens = usage.prompt_tokens if usage and usage.prompt_tokens is not None else 0
        completion_tokens = usage.completion_tokens if usage and usage.completion_tokens is not None else 0
        total_tokens = (
            usage.total_tokens if usage and usage.total_tokens is not None else prompt_tokens + completion_tokens
        )
        details = PromptTokensDetailsWrapper(web_search_requests=web_search_requests) if web_search_requests else None
        usage_obj = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=details,
        )
        if web_search_premium_requests:
            setattr(usage_obj, "web_search_premium_requests", web_search_premium_requests)
        return usage_obj

    @staticmethod
    def _finish_reason(request_data: dict, usage: Optional[MistralConversationUsage]) -> str:
        """The Conversations API returns no finish/stop reason, so infer truncation
        from the token budget: ``length`` when the completion filled the requested
        ``max_tokens``, otherwise ``stop``."""
        args = STR_OBJ_DICT.validate_python(request_data.get("completion_args") or {})
        max_tokens = args.get("max_tokens")
        completion_tokens = usage.completion_tokens if usage else None
        if isinstance(max_tokens, int) and isinstance(completion_tokens, int) and completion_tokens >= max_tokens:
            return "length"
        return "stop"

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: object,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        logging_obj.post_call(original_response=raw_response.text)
        logging_obj.model_call_details["response_headers"] = raw_response.headers

        parsed = MistralConversationsResponse.model_validate(raw_response.json())
        text, annotations = self._extract_message(parsed.outputs)
        standard_searches, premium_searches = self._count_web_searches(parsed.usage, parsed.outputs)

        message = Message(
            role="assistant",
            content=text or None,
            annotations=annotations or None,
        )
        model_response.choices = [
            Choices(index=0, message=message, finish_reason=self._finish_reason(request_data, parsed.usage))
        ]
        model_response.model = model
        if parsed.conversation_id:
            model_response.id = parsed.conversation_id
        setattr(
            model_response,
            "usage",
            self._build_usage(parsed.usage, standard_searches + premium_searches, premium_searches),
        )
        return model_response
