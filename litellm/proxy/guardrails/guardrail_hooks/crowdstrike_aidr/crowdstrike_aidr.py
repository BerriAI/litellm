import json
import os
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    NamedTuple,
    Optional,
    Union,
    cast,
)

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Any, override

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.llms.openai import AllMessageValues, OpenAIChatCompletionToolParam
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailMissingSecrets(Exception):
    """Custom exception for missing CrowdStrike AIDR secrets."""

    pass


class _TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class _ImageUrl(BaseModel):
    url: str


class _ImageUrlContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    image_url: _ImageUrl


_ContentPart = Annotated[Union[_TextContentPart, _ImageUrlContentPart], Field(discriminator="type")]


class _Message(BaseModel):
    role: str
    content: Optional[Union[str, list[_ContentPart]]] = None


class _GuardInput(BaseModel):
    messages: list[_Message]
    tools: Optional[Sequence[OpenAIChatCompletionToolParam]] = None


class _GuardChatCompletionsResult(BaseModel):
    guard_output: Optional[_GuardInput] = None
    """Updated structured prompt."""
    blocked: Optional[bool] = None
    """Whether or not the prompt triggered a block detection."""
    transformed: Optional[bool] = None
    """Whether or not the original input was transformed."""
    detectors: Optional[dict[str, Any]] = None
    """Result of the policy analyzing and input prompt."""


class _GuardChatCompletionsResponse(BaseModel):
    result: Optional[_GuardChatCompletionsResult] = None


class _FilteredMessages(NamedTuple):
    """Subset of a conversation selected for guardrail analysis."""

    messages: list[AllMessageValues]
    """Messages subset."""
    indices: tuple[int, ...]
    """Positions of the subset's messages in the original list."""


class _GuardInputWithIndices(NamedTuple):
    guard_input: _GuardInput
    """Guard API payload."""
    sent_indices: tuple[int, ...]
    """Positions of the guard input's messages in the original list."""


def _normalize_content(raw: object) -> str | list[_ContentPart] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return json.dumps(raw)
    parts: list[_ContentPart] = []
    for block in raw:
        if not isinstance(block, dict):
            parts.append(_TextContentPart(text=json.dumps(block)))
            continue

        t = block.get("type")
        if t == "text" and isinstance(block.get("text"), str):
            parts.append(_TextContentPart(text=cast(str, block["text"])))
        elif t == "image_url":
            iu = block.get("image_url")
            url = iu if isinstance(iu, str) else str((iu or {}).get("url", ""))
            parts.append(_ImageUrlContentPart(image_url=_ImageUrl(url=url)))

        # Any other types are not recognized by the CrowdStrike AIDR API.

    return parts


def _extract_text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(parts)
    return ""


def _extract_text_from_message(message: _Message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "\n".join(part.text for part in content if isinstance(part, _TextContentPart))


def _merge_metadata_bags(request_data: Mapping[str, Any]) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    present = False
    for bag in (request_data.get("metadata"), request_data.get("litellm_metadata")):
        if isinstance(bag, Mapping):
            present = True
            merged.update(bag)
    return merged if present else None


def _messages_since_last_assistant(
    messages: list[AllMessageValues],
) -> _FilteredMessages:
    if not messages:
        return _FilteredMessages([], ())

    if messages[-1]["role"] == "assistant":
        indices = tuple(i for i, m in enumerate(messages) if m["role"] == "system") + (len(messages) - 1,)
        return _FilteredMessages([messages[i] for i in indices], indices)

    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    system_indices = tuple(i for i in range(last_assistant_idx + 1) if messages[i]["role"] == "system")
    tail_indices = tuple(range(last_assistant_idx + 1, len(messages)))
    indices = system_indices + tail_indices
    return _FilteredMessages([messages[i] for i in indices], indices)


def _merge_request_transforms(
    guard_output: _GuardInput,
    structured_messages: list[AllMessageValues] | None,
    texts: list[str],
    sent_indices: tuple[int, ...],
) -> list[str]:
    returned_texts = [_extract_text_from_message(msg) for msg in guard_output.messages]
    original_texts = (
        [_extract_text_from_content(m.get("content")) for m in structured_messages] if structured_messages else texts
    )
    replacements = {
        idx: returned_texts[pos]
        for pos, idx in enumerate(sent_indices)
        if pos < len(returned_texts) and idx < len(original_texts)
    }
    return [replacements.get(idx, original) for idx, original in enumerate(original_texts)]


def _apply_message_redaction(original: AllMessageValues, redacted: _Message) -> AllMessageValues:
    content = original.get("content")
    if isinstance(content, str):
        return cast(AllMessageValues, {**original, "content": _extract_text_from_message(redacted)})
    if isinstance(content, list) and _extract_text_from_content(content):
        redacted_content = redacted.content
        new_content = (
            [part.model_dump() for part in redacted_content] if isinstance(redacted_content, list) else redacted_content
        )
        return cast(AllMessageValues, {**original, "content": new_content})
    return original


def _redacted_messages(
    processed_messages: list[AllMessageValues],
    guard_output: _GuardInput,
    sent_indices: tuple[int, ...],
    full_messages: list[AllMessageValues],
) -> list[AllMessageValues] | None:
    redactions = {
        id(processed_messages[idx]): _apply_message_redaction(processed_messages[idx], guard_output.messages[pos])
        for pos, idx in enumerate(sent_indices)
        if pos < len(guard_output.messages) and idx < len(processed_messages)
    }
    if not redactions.keys() <= {id(message) for message in full_messages}:
        return None
    return [redactions.get(id(message), message) for message in full_messages]


class CrowdStrikeAIDRHandler(CustomGuardrail):
    """
    CrowdStrike AIDR AI Guardrail handler to interact with the CrowdStrike AIDR
    AI Guard service.
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the CrowdStrikeAIDRHandler.

        Args:
            guardrail_name (str): The name of the guardrail instance.
            api_key (str | None): The CrowdStrike AIDR API key. Reads from CS_AIDR_TOKEN env var if None.
            api_base (str | None): The CrowdStrike AIDR API base URL. Reads from CS_AIDR_BASE_URL env var if None.
            **kwargs: Additional arguments passed to the CustomGuardrail base class.
        """
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        self.api_key = api_key or os.environ.get("CS_AIDR_TOKEN")
        if not self.api_key:
            raise CrowdStrikeAIDRGuardrailMissingSecrets(
                "CrowdStrike AIDR API Key not found. Set CS_AIDR_TOKEN environment variable or pass it in litellm_params."
            )

        self.api_base = api_base or os.environ.get("CS_AIDR_BASE_URL")
        if not self.api_base:
            raise CrowdStrikeAIDRGuardrailMissingSecrets(
                "CrowdStrike AIDR API base URL is required. Set CS_AIDR_BASE_URL environment variable or pass it in litellm_params."
            )

        # Pass relevant kwargs to the parent class
        super().__init__(guardrail_name=guardrail_name, **kwargs)
        verbose_proxy_logger.debug(
            f"Initialized CrowdStrike AIDR Guardrail: name={guardrail_name}, api_base={self.api_base}"
        )

    async def _call_crowdstrike_aidr_guard(
        self, payload: dict[str, Any], hook_name: str
    ) -> _GuardChatCompletionsResult:
        """
        Makes the API call to the CrowdStrike AIDR AI Guard endpoint.
        The function itself will raise an error if a response should be blocked,
        but otherwise will return a list of redacted messages that the caller
        should act on.

        Args:
            payload (dict): The request payload.
            hook_name (str): Name of the hook calling this function (for logging).

        Raises:
            HTTPException: If the CrowdStrike AIDR API returns a 'blocked: true' response.
            Exception: For other API call failures.

        Returns:
            The parsed `result` body of the API response.
        """
        endpoint = f"{self.api_base}/v1/guard_chat_completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        verbose_proxy_logger.debug(
            f"CrowdStrike AIDR Guardrail ({hook_name}): Calling endpoint {endpoint} with payload: {payload}"
        )

        response = await self.async_handler.post(url=endpoint, json=payload, headers=headers)
        assert response is not None
        response.raise_for_status()

        result = _GuardChatCompletionsResponse.model_validate(response.json()).result or _GuardChatCompletionsResult()

        if result.blocked:
            verbose_proxy_logger.warning(
                f"CrowdStrike AIDR Guardrail ({hook_name}): Request blocked. Response: {result}"
            )
            raise HTTPException(
                status_code=400,  # Bad Request, indicating violation
                detail={
                    "error": "Violated CrowdStrike AIDR guardrail policy",
                    "guardrail_name": self.guardrail_name,
                },
            )
        verbose_proxy_logger.debug(
            f"CrowdStrike AIDR Guardrail ({hook_name}): Request passed. Response: {result.detectors}"
        )

        return result

    def _build_guard_input_for_request(self, inputs: GenericGuardrailAPIInputs) -> _GuardInputWithIndices | None:
        guard_input = _GuardInput(messages=[], tools=[])
        structured_messages = inputs.get("structured_messages")
        texts = inputs.get("texts", [])
        tools = inputs.get("tools")

        if structured_messages:
            filtered = _messages_since_last_assistant(structured_messages)
            for message in filtered.messages:
                content = _normalize_content(message.get("content"))
                if content is None or len(content) == 0:
                    content = ""
                guard_input.messages.append(_Message(role=message["role"], content=content))
            indices = filtered.indices
        elif texts:
            guard_input.messages = [_Message(role="user", content=text) for text in texts]
            indices = tuple(range(len(texts)))
        else:
            verbose_proxy_logger.warning("CrowdStrike AIDR Guardrail: No messages or texts provided for input request")
            return None

        if tools:
            guard_input.tools = tools

        return _GuardInputWithIndices(guard_input, indices)

    def _build_guard_input_for_response(self, inputs: GenericGuardrailAPIInputs) -> _GuardInput:
        output_texts: list[str] = inputs.get("texts", [])
        return _GuardInput(
            messages=[_Message(role="assistant", content=text) for text in output_texts],
            tools=inputs.get("tools", []),
        )

    def _extract_transformed_texts(self, guard_output: _GuardInput, num_assistant_messages: int) -> list[str]:
        tail = guard_output.messages[-num_assistant_messages:] if num_assistant_messages > 0 else []
        return [_extract_text_from_message(msg) for msg in tail]

    def _writeback_messages(
        self,
        structured_messages: list[AllMessageValues],
        guard_output: _GuardInput,
        sent_indices: tuple[int, ...],
        request_data: dict,
    ) -> list[AllMessageValues] | None:
        if effective_skip_system_message_for_guardrail(self) or effective_skip_tool_message_for_guardrail(self):
            request_messages = request_data.get("messages")
            full_messages = (
                cast("list[AllMessageValues]", request_messages)
                if isinstance(request_messages, list)
                else structured_messages
            )
        else:
            full_messages = structured_messages
        return _redacted_messages(structured_messages, guard_output, sent_indices, full_messages)

    @log_guardrail_information
    @override
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        verbose_proxy_logger.debug(f"CrowdStrike AIDR Guardrail: Applying guardrail to {input_type}")

        # Extract inputs
        texts = inputs.get("texts", [])
        structured_messages = inputs.get("structured_messages")
        tools = inputs.get("tools")
        tool_calls = inputs.get("tool_calls")

        # Build guard_input based on input_type
        sent_indices: tuple[int, ...] = ()
        if input_type == "request":
            request_result = self._build_guard_input_for_request(inputs)
            if request_result is None:
                return inputs
            guard_input = request_result.guard_input
            sent_indices = request_result.sent_indices
            event_type = "input"
            hook_name = "apply_guardrail (request)"
        else:
            guard_input = self._build_guard_input_for_response(inputs)
            if len(guard_input.messages) == 0:
                return inputs
            event_type = "output"
            hook_name = "apply_guardrail (response)"

        ai_guard_payload: dict[str, Any] = {
            "guard_input": guard_input.model_dump(mode="json"),
            "event_type": event_type,
        }

        model = inputs.get("model")
        if model:
            ai_guard_payload["model"] = model

        metadata = _merge_metadata_bags(request_data)
        if metadata is not None:
            user_id = metadata.get("user_api_key_user_id")
            if user_id:
                ai_guard_payload["user_id"] = user_id

            extra_info: dict[str, str] = {}
            user_email = metadata.get("user_api_key_user_email")
            if user_email:
                extra_info["user_name"] = user_email
            ai_guard_payload["extra_info"] = extra_info

        result = await self._call_crowdstrike_aidr_guard(ai_guard_payload, hook_name)

        if "body" in request_data or "messages" in request_data:
            add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)

        if not result.transformed or result.guard_output is None:
            return inputs

        guard_output = result.guard_output

        if input_type == "request":
            transformed_texts = _merge_request_transforms(guard_output, structured_messages, texts, sent_indices)
        else:
            transformed_texts = self._extract_transformed_texts(guard_output, len(texts))

        result_inputs: GenericGuardrailAPIInputs = {"texts": transformed_texts}
        if tools:
            result_inputs["tools"] = tools
        if tool_calls:
            result_inputs["tool_calls"] = tool_calls
        if structured_messages:
            rebuilt = (
                self._writeback_messages(structured_messages, guard_output, sent_indices, request_data)
                if input_type == "request"
                else None
            )
            result_inputs["structured_messages"] = rebuilt if rebuilt is not None else structured_messages

        return result_inputs

    @override
    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.crowdstrike_aidr import (
            CrowdStrikeAIDRGuardrailConfigModel,
        )

        return CrowdStrikeAIDRGuardrailConfigModel
