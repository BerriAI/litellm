from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Optional

from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    scoped_structured_message_indices,
)
from litellm.types.llms.openai import (
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionToolCallFunctionChunk,
)

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import (
        CustomGuardrail,
        ModifyResponseException,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import (
        AllMessageValues,
        ChatCompletionToolCallChunk,
        ChatCompletionToolParam,
    )
    from litellm.types.utils import GenericGuardrailAPIInputs


@dataclass(slots=True)
class StreamTransformSink:
    """Out-parameter used by ``process_output_streaming_response`` to hand the
    guardrailed streaming state back to the caller.

    The streaming text-transform path must not mutate ``responses_so_far`` (it is
    the raw accumulator the guardrail re-reads every round), so the guardrailed
    accumulated text per choice (``mutated_text_per_choice``, keyed by
    ``StreamingChoices.index``) and the per-choice trailing holdback the guardrail
    requested (``holdback_per_choice``, from ``stream_holdback_chars``) are
    reported here instead of in place. Only the OpenAI chat handler populates this
    today; the hook passes a fresh sink per round and reads it afterwards. A
    mutable dataclass is deliberate: it is a write-once output parameter for a
    single call, not shared state.
    """

    mutated_text_per_choice: dict[int, str] = field(default_factory=dict)
    holdback_per_choice: dict[int, int] = field(default_factory=dict)


class BaseTranslation(ABC):
    @staticmethod
    def transform_user_api_key_dict_to_metadata(
        user_api_key_dict: Any | None,
    ) -> dict[str, Any]:
        """
        Transform user_api_key_dict to a metadata dict with prefixed keys.

        Converts keys like 'user_id' to 'user_api_key_user_id' to clearly indicate
        the source of the metadata.

        Args:
            user_api_key_dict: UserAPIKeyAuth object or dict with user information

        Returns:
            Dict with keys prefixed with 'user_api_key_'
        """
        if user_api_key_dict is None:
            return {}

        # Convert to dict if it's a Pydantic object
        user_dict = user_api_key_dict.model_dump() if hasattr(user_api_key_dict, "model_dump") else user_api_key_dict

        if not isinstance(user_dict, dict):
            return {}

        # Transform keys to be prefixed with 'user_api_key_'
        transformed: Final = {}
        for key, value in user_dict.items():
            # Skip None values and internal fields
            if value is None or key.startswith("_"):
                continue

            # If key already has the prefix, use as-is, otherwise add prefix
            if key.startswith("user_api_key_"):
                transformed[key] = value
            else:
                transformed[f"user_api_key_{key}"] = value

        return transformed

    @abstractmethod
    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        """
        Process input messages with guardrails.

        Note: user_api_key_dict metadata should be available in the data dict.
        """

    @abstractmethod
    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: dict | None = None,
    ) -> Any:
        """
        Process output response with guardrails.

        Args:
            response: The response object from the LLM
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata (passed separately since response doesn't contain it)
        """

    async def process_output_streaming_response(
        self,
        responses_so_far: list[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: dict | None = None,
        stream_transform_sink: StreamTransformSink | None = None,
    ) -> Any:
        """
        Process output streaming response with guardrails.

        Optional to override in subclasses. ``stream_transform_sink`` is the
        out-parameter used by handlers that support streaming text
        transformations (see ``StreamTransformSink``); base handlers ignore it.
        """
        return responses_so_far

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: list[Any] | None = None,
    ) -> list[bytes] | None:
        """
        Build the streaming chunks that deliver a guardrail block message and
        cleanly terminate the stream in this provider's wire format.

        ``stream_started`` is True when real chunks were already sent to the
        client: the result must *continue* the in-progress message (e.g. close
        the open content block and append the block message) rather than start
        a new one, which clients reject. ``responses_so_far`` provides the prior
        chunks needed to do so. When False, nothing has been sent and a
        standalone block message is emitted.

        Returns None when the format has no safe terminator; the caller then
        re-raises ``exc`` so the proxy can surface a clean error instead.
        Override in provider subclasses that support synthesizing a block
        stream.
        """
        return None

    def get_structured_messages(self, data: dict) -> list["AllMessageValues"] | None:
        """
        Convert request data to OpenAI-spec structured messages.

        Override in subclasses for format-specific conversion.

        Returns None if no convertible content is found.
        """
        return None

    def scoped_request_conversation(
        self,
        request_data: dict,  # mutable-ok: API request payload
        guardrail_to_apply: "CustomGuardrail",
    ) -> tuple["AllMessageValues", ...] | None:
        """
        The request conversation as the guardrail's request scan saw it: the
        handler's structured messages with the operator scoping flags applied.

        Override when the request scan scopes differently (e.g. Anthropic's
        top-level system prompt hoisting).
        """
        structured_messages: Final = self.get_structured_messages(request_data)
        if not structured_messages:
            return None
        scoped_indices: Final = scoped_structured_message_indices(
            structured_messages,
            scan_only_tool_results=effective_scan_only_tool_results_for_guardrail(guardrail_to_apply),
            skip_system=effective_skip_system_message_for_guardrail(guardrail_to_apply),
            skip_tool=effective_skip_tool_message_for_guardrail(guardrail_to_apply),
        )
        return tuple(structured_messages[index] for index in scoped_indices) or None

    def response_scan_conversation(
        self,
        request_data: dict | None,  # mutable-ok: API request payload
        guardrail_to_apply: "CustomGuardrail",
        response_turns: Sequence["AllMessageValues"],
    ) -> tuple["AllMessageValues", ...] | None:
        """
        Full conversation for a response scan: the scoped request conversation
        with the model's response turns appended.

        Returns None when the request context is unavailable (SDK/direct-call
        path fabricates request_data without messages) or when nothing was
        extracted from the response; guardrails then fall back to scanning the
        extracted texts and tool calls, which must not be shadowed by a
        conversation that lacks a response turn.
        """
        if request_data is None or not response_turns:
            return None
        request_conversation: Final = self.scoped_request_conversation(request_data, guardrail_to_apply)
        if request_conversation is None:
            return None
        return (*request_conversation, *response_turns)

    def attach_response_scan_context(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict | None,  # mutable-ok: API request payload
        guardrail_to_apply: "CustomGuardrail",
        response_turns: Sequence["AllMessageValues"],
    ) -> None:
        """
        Put the response-scan conversation and the request's tools on ``inputs``
        when the request context allows building them; no-op otherwise.
        """
        structured_conversation: Final = self.response_scan_conversation(
            request_data, guardrail_to_apply, response_turns
        )
        if structured_conversation is None or request_data is None:
            return
        inputs["structured_messages"] = list(structured_conversation)  # rebind-ok: out-param; field type is list
        response_scan_tools: Final = self.request_tools_for_guardrail(request_data, guardrail_to_apply)
        if response_scan_tools:
            inputs["tools"] = list(response_scan_tools)  # rebind-ok: out-parameter; the field type is a list

    def request_tools_for_guardrail(
        self,
        request_data: dict,  # mutable-ok: API request payload
        guardrail_to_apply: "CustomGuardrail",
    ) -> tuple["ChatCompletionToolParam", ...] | None:
        """
        The request's tool definitions in the shape the request scan sends them.

        Override in tool-capable handlers; default returns None.
        """
        return None

    @staticmethod
    def assistant_tool_call(
        tool_call_id: str | None,
        name: str | None,
        arguments: str,
    ) -> ChatCompletionAssistantToolCall:
        """One assistant-message tool call in the OpenAI wire shape."""
        return ChatCompletionAssistantToolCall(
            id=tool_call_id,
            type="function",
            function=ChatCompletionToolCallFunctionChunk(name=name, arguments=arguments),
        )

    @staticmethod
    def assistant_turn_from_extraction(
        texts: Sequence[str],
        tool_calls: Sequence["ChatCompletionAssistantToolCall | ChatCompletionToolCallChunk"] | None = None,
    ) -> tuple["ChatCompletionAssistantMessage", ...]:
        """
        One OpenAI-shape assistant turn built from the texts and tool calls a
        handler's response extraction collected; empty when there is nothing.
        Tool calls are normalized to the assistant-message shape, dropping
        extraction-only fields such as ``index``.
        """
        tool_call_items: Final = tuple(
            BaseTranslation.assistant_tool_call(
                tool_call_id=item.get("id"),
                name=item["function"].get("name"),
                arguments=item["function"].get("arguments") or "",
            )
            for item in tool_calls or ()
        )
        if not texts and not tool_call_items:
            return ()
        if tool_call_items:
            turn_with_tools: Final[ChatCompletionAssistantMessage] = {
                "role": "assistant",
                "content": "\n".join(texts),
                "tool_calls": list(tool_call_items),  # mutable-ok: the message field type is a list
            }
            return (turn_with_tools,)
        turn: Final[ChatCompletionAssistantMessage] = {"role": "assistant", "content": "\n".join(texts)}
        return (turn,)

    def extract_request_tool_names(self, data: dict) -> list[str]:
        """
        Extract tool names from the request body for allowlist/policy checks.
        Override in tool-capable handlers; default returns [].
        """
        return []
