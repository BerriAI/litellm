"""
Reproduction script for https://github.com/BerriAI/litellm/issues/14633

The auto-router fails when messages contain tool calls because:
1. Function signatures use Dict[str, str] but tool call messages have non-string values
   (tool_calls is a list of dicts, content can be None)
2. The content extraction in auto_router.py doesn't handle None content from
   assistant messages with tool_calls

Run with: uv run python repro_14633.py
"""

import sys

from litellm.types.router import PreRoutingHookResponse


def test_pre_routing_hook_response_with_tool_calls():
    """
    The PreRoutingHookResponse model was fixed to use Dict[str, Any].
    This test confirms it no longer rejects messages with tool_calls.
    """
    messages_with_tool_calls = [
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "The weather in NYC is 72°F and sunny.",
        },
        {"role": "user", "content": "Now tell me about London"},
    ]

    response = PreRoutingHookResponse(
        model="test-model",
        messages=messages_with_tool_calls,
    )
    print("[PASS] PreRoutingHookResponse accepts messages with tool_calls (Dict[str, Any])")
    print(f"  model={response.model}, num_messages={len(response.messages)}")


def test_content_extraction_fixed():
    """
    Verifies the content extraction fix in auto_router.py.

    Previously, when the last message was an assistant message with tool_calls
    and content=None, the code would pass None to the semantic router.

    Now _extract_text_from_messages finds the last *user* message and handles
    None content and multimodal (list) content correctly.
    """
    from litellm.router_strategy.auto_router.auto_router import AutoRouter

    messages_ending_with_assistant_tool_call = [
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
    ]

    message_content = AutoRouter._extract_text_from_messages(
        messages_ending_with_assistant_tool_call
    )

    print(f"\n[CHECK] Content extraction from conversation with tool_calls:")
    print(f"  _extract_text_from_messages result = {repr(message_content)}")

    if message_content == "What's the weather in NYC?":
        print("  --> FIXED: correctly extracts last user message")
        return False
    else:
        print(f"  --> BUG: expected 'What's the weather in NYC?', got {repr(message_content)}")
        return True


def test_type_annotation_mismatch():
    """
    Shows that function signatures still declare messages as Dict[str, str]
    even though messages with tool_calls have non-string values.

    This is a type-hint issue (not a runtime error), but it causes static
    analysis tools and IDEs to flag valid tool-call messages as type errors.
    """
    import inspect

    # Check the auto_router function signature
    from litellm.integrations.custom_logger import CustomLogger

    sig = inspect.signature(CustomLogger.async_pre_routing_hook)
    messages_param = sig.parameters["messages"]
    annotation = str(messages_param.annotation)

    print(f"\n[INFO] Type annotation check:")
    print(f"  CustomLogger.async_pre_routing_hook 'messages' param: {annotation}")

    if "Dict[str, str]" in annotation:
        print("  --> ISSUE: Still uses Dict[str, str], should be Dict[str, Any]")
        print("  This means tool_call messages (with list/dict values) violate the type contract")
        return True
    else:
        print("  --> OK: Uses Dict[str, Any]")
        return False


def test_multimodal_content_fixed():
    """
    Verifies that multimodal messages (content as a list of blocks) are
    handled correctly by _extract_text_from_messages.
    """
    from litellm.router_strategy.auto_router.auto_router import AutoRouter

    messages_with_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        }
    ]

    message_content = AutoRouter._extract_text_from_messages(messages_with_image)

    print(f"\n[CHECK] Content extraction from multimodal message:")
    print(f"  type(message_content) = {type(message_content).__name__}")
    print(f"  message_content = {repr(message_content)}")

    if isinstance(message_content, str) and message_content == "What's in this image?":
        print("  --> FIXED: correctly extracts text from multimodal content")
        return False
    else:
        print(f"  --> BUG: expected string, got {type(message_content).__name__}")
        return True


def main():
    print("=" * 70)
    print("Reproduction for GitHub Issue #14633")
    print("AutoRouter fails with tool call messages")
    print("=" * 70)

    bugs_found = 0

    # Test 1: PreRoutingHookResponse type (partially fixed)
    test_pre_routing_hook_response_with_tool_calls()

    # Test 2: Content extraction fix (None content)
    if test_content_extraction_fixed():
        bugs_found += 1

    # Test 3: Type annotation fix
    if test_type_annotation_mismatch():
        bugs_found += 1

    # Test 4: Multimodal content fix
    if test_multimodal_content_fixed():
        bugs_found += 1

    print("\n" + "=" * 70)
    if bugs_found == 0:
        print("ALL CHECKS PASSED - all bugs have been fixed!")
    else:
        print(f"Results: {bugs_found} bug(s) remaining")
    print("=" * 70)

    sys.exit(1 if bugs_found > 0 else 0)


if __name__ == "__main__":
    main()
