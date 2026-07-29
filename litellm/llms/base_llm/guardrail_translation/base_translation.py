from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import (
        CustomGuardrail,
        ModifyResponseException,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import AllMessageValues


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
    ) -> Dict[str, Any]:
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
        transformed = {}
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
        pass

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
        pass

    async def process_output_streaming_response(
        self,
        responses_so_far: List[Any],
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

    def get_structured_messages(self, data: dict) -> List["AllMessageValues"] | None:
        """
        Convert request data to OpenAI-spec structured messages.

        Override in subclasses for format-specific conversion.

        Returns None if no convertible content is found.
        """
        return None

    def extract_request_tool_names(self, data: dict) -> List[str]:
        """
        Extract tool names from the request body for allowlist/policy checks.
        Override in tool-capable handlers; default returns [].
        """
        return []
