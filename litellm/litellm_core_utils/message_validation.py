"""
Pre-flight validation for LLM message sequences.

Validates message sequences before sending to LLM providers to catch common
issues that would cause API errors. This saves API costs and provides clearer
error messages than provider-side validation.

Validates:
- Tool call / tool response pairing (no orphans, no duplicates)
- Message role sequencing rules
- Provider-specific constraints

Example usage:
    from litellm.litellm_core_utils.message_validation import validate_messages

    messages = [
        {"role": "user", "content": "Run ls"},
        {"role": "assistant", "tool_calls": [{"id": "tc1", ...}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "file.txt"},
    ]

    errors = validate_messages(messages, provider="openai")
    if errors:
        raise ValueError(f"Invalid messages: {errors}")
"""

from typing import Any, Literal

# Provider types supported by validation
ProviderType = Literal["openai", "anthropic", "auto"]


def validate_messages(
    messages: list[dict[str, Any]],
    provider: ProviderType = "auto",
    tools_defined: bool = False,
) -> list[str]:
    """Validate a message sequence for LLM API compatibility.

    Checks for common issues that cause API errors:
    - Tool calls without matching tool responses
    - Duplicate tool responses for the same tool_call_id
    - Invalid message ordering

    Args:
        messages: List of message dictionaries in OpenAI or Anthropic format
        provider: Target provider ("openai", "anthropic", or "auto" to detect)
        tools_defined: Whether tools are defined in the request

    Returns:
        List of validation error messages (empty if valid)
    """
    if not messages:
        return []

    # Auto-detect provider from message format
    if provider == "auto":
        provider = _detect_provider(messages)

    if provider == "anthropic":
        return _validate_anthropic_messages(messages, tools_defined)
    else:
        return _validate_openai_messages(messages, tools_defined)


def _detect_provider(messages: list[dict[str, Any]]) -> Literal["openai", "anthropic"]:
    """Detect provider from message format."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type in ("tool_use", "tool_result"):
                        return "anthropic"
    return "openai"


def _validate_openai_messages(
    messages: list[dict[str, Any]],
    tools_defined: bool,
) -> list[str]:
    """Validate OpenAI Chat Completions message format.

    OpenAI rules:
    - Each tool_call in an assistant message must have exactly one
      corresponding tool message with matching tool_call_id
    - Tool messages must follow the assistant message containing the tool_call
    - No duplicate tool_call_ids in tool responses
    """
    errors: list[str] = []

    # Track tool_call_ids and their responses
    pending_tool_calls: dict[str, int] = {}  # tool_call_id -> message index
    seen_tool_responses: set[str] = set()

    for i, msg in enumerate(messages):
        role = msg.get("role")

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    if tc_id in pending_tool_calls:
                        errors.append(
                            f"Duplicate tool_call id '{tc_id}' in assistant messages"
                        )
                    pending_tool_calls[tc_id] = i

        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                if tc_id in seen_tool_responses:
                    errors.append(
                        f"Duplicate tool response for tool_call_id '{tc_id}'"
                    )
                seen_tool_responses.add(tc_id)

                if tc_id in pending_tool_calls:
                    del pending_tool_calls[tc_id]
                else:
                    errors.append(
                        f"Tool response for unknown tool_call_id '{tc_id}' "
                        f"(no matching tool_call found)"
                    )

        elif role in ("user", "system", "developer"):
            # User/system messages after tool_calls but before tool responses
            # indicate missing tool responses
            if pending_tool_calls and tools_defined:
                unresolved = list(pending_tool_calls.keys())
                errors.append(
                    f"Unresolved tool_calls before {role} message: {unresolved}. "
                    f"Each tool_call must have a matching tool response."
                )

    # Check for any remaining unresolved tool calls at end of messages
    if pending_tool_calls and tools_defined:
        unresolved = list(pending_tool_calls.keys())
        errors.append(
            f"Unresolved tool_calls at end of messages: {unresolved}. "
            f"Each tool_call must have a matching tool response."
        )

    return errors


def _validate_anthropic_messages(
    messages: list[dict[str, Any]],
    tools_defined: bool,
) -> list[str]:
    """Validate Anthropic Messages API format.

    Anthropic rules:
    - Each tool_use block must have exactly one tool_result block
      with matching id in a subsequent user message
    - tool_result must come in the user message immediately following
      the assistant message with tool_use
    - No duplicate tool_use_ids in tool_result blocks
    """
    errors: list[str] = []

    # Track tool_use ids and their results
    pending_tool_uses: dict[str, int] = {}  # tool_use_id -> message index
    seen_tool_results: set[str] = set()

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tu_id = block.get("id")
                    if tu_id:
                        if tu_id in pending_tool_uses:
                            errors.append(
                                f"Duplicate tool_use id '{tu_id}' in assistant messages"
                            )
                        pending_tool_uses[tu_id] = i

        elif role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tu_id = block.get("tool_use_id")
                    if tu_id:
                        if tu_id in seen_tool_results:
                            errors.append(
                                f"Duplicate tool_result for tool_use_id '{tu_id}'"
                            )
                        seen_tool_results.add(tu_id)

                        if tu_id in pending_tool_uses:
                            del pending_tool_uses[tu_id]
                        else:
                            errors.append(
                                f"tool_result for unknown tool_use_id '{tu_id}' "
                                f"(no matching tool_use found)"
                            )

        elif role == "user" and not isinstance(content, list):
            # Plain user message after tool_use but before tool_result
            if pending_tool_uses and tools_defined:
                unresolved = list(pending_tool_uses.keys())
                errors.append(
                    f"Unresolved tool_use blocks before user message: {unresolved}. "
                    f"Each tool_use must have a matching tool_result."
                )

    # Check for any remaining unresolved tool uses at end of messages
    if pending_tool_uses and tools_defined:
        unresolved = list(pending_tool_uses.keys())
        errors.append(
            f"Unresolved tool_use blocks at end of messages: {unresolved}. "
            f"Each tool_use must have a matching tool_result."
        )

    return errors


def validate_responses_input(
    input_items: list[dict[str, Any]],
    tools_defined: bool = False,
) -> list[str]:
    """Validate OpenAI Responses API input format.

    Responses API rules:
    - Each function_call must have exactly one function_call_output
      with matching call_id
    - No duplicate call_ids in function_call_output items
    """
    errors: list[str] = []

    pending_calls: dict[str, int] = {}  # call_id -> item index
    seen_outputs: set[str] = set()

    for i, item in enumerate(input_items):
        item_type = item.get("type")

        if item_type == "function_call":
            call_id = item.get("call_id")
            if call_id:
                if call_id in pending_calls:
                    errors.append(f"Duplicate function_call id '{call_id}'")
                pending_calls[call_id] = i

        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if call_id:
                if call_id in seen_outputs:
                    errors.append(
                        f"Duplicate function_call_output for call_id '{call_id}'"
                    )
                seen_outputs.add(call_id)

                if call_id in pending_calls:
                    del pending_calls[call_id]
                else:
                    errors.append(
                        f"function_call_output for unknown call_id '{call_id}'"
                    )

    if pending_calls and tools_defined:
        unresolved = list(pending_calls.keys())
        errors.append(
            f"Unresolved function_calls: {unresolved}. "
            f"Each function_call must have a matching function_call_output."
        )

    return errors
