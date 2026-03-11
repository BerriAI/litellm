from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history


def test_thought_true_creates_thinking_block():
    """
    Test that a part with thought=True and non-empty text creates a thinking block.
    Per Google's docs, parts must have thought=True to be thinking content.
    """
    parts = [{"text": "Some thinking", "thought": True, "thoughtSignature": "sig-1"}]
    config = VertexGeminiConfig()
    thinking_blocks = config._extract_thinking_blocks_from_parts(parts)
    assert len(thinking_blocks) == 1
    block = thinking_blocks[0]
    assert block["thinking"] == "Some thinking"
    assert block["signature"] == "sig-1"


def test_thought_true_with_empty_text_creates_block():
    """
    Test that a part with thought=True but empty text still creates a thinking block.
    """
    parts = [{"text": "", "thought": True, "thoughtSignature": "sig-2"}]
    config = VertexGeminiConfig()
    thinking_blocks = config._extract_thinking_blocks_from_parts(parts)
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["thinking"] == ""


def test_thought_signature_without_thought_does_not_create_block():
    """
    Test that a part with thoughtSignature but without thought=True does NOT create
    a thinking block. Per Google's docs, thoughtSignature is for multi-turn context
    preservation and does not indicate that the content is thinking.
    """
    parts = [{"text": "Some text", "thoughtSignature": "sig-3"}]
    config = VertexGeminiConfig()
    thinking_blocks = config._extract_thinking_blocks_from_parts(parts)
    assert thinking_blocks == []


def test_extract_thought_signatures_from_regular_parts():
    """
    Test that thoughtSignatures are extracted from regular text parts (without thought=True).
    This is the key feature for Gemini 3 multi-turn context preservation.
    """
    parts = [{"text": "I am Gemini", "thoughtSignature": "sig-regular-123"}]
    config = VertexGeminiConfig()
    
    # Should NOT create thinking block
    thinking_blocks = config._extract_thinking_blocks_from_parts(parts)
    assert thinking_blocks == []
    
    # Should extract thought signature
    signatures = config._extract_thought_signatures_from_parts(parts)
    assert signatures is not None
    assert len(signatures) == 1
    assert signatures[0] == "sig-regular-123"


def test_extract_multiple_thought_signatures():
    """
    Test extraction of multiple thoughtSignatures from different parts.
    """
    parts = [
        {"text": "Part 1", "thoughtSignature": "sig-1"},
        {"text": "Part 2", "thoughtSignature": "sig-2"},
        {"text": "Part 3"}  # No signature
    ]
    config = VertexGeminiConfig()
    signatures = config._extract_thought_signatures_from_parts(parts)
    
    assert signatures is not None
    assert len(signatures) == 2
    assert signatures[0] == "sig-1"
    assert signatures[1] == "sig-2"


def test_round_trip_thought_signature_in_conversation():
    """
    Test that thoughtSignatures are properly round-tripped through conversation history.
    This ensures multi-turn context preservation works correctly.
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hi there",
            "provider_specific_fields": {
                "thought_signatures": ["sig-round-trip-abc"]
            }
        },
        {"role": "user", "content": "How are you?"}
    ]
    
    gemini_contents = _gemini_convert_messages_with_history(messages)
    
    # Find the assistant (model) message
    model_message = None
    for content in gemini_contents:
        if content.get("role") == "model":
            model_message = content
            break
    
    assert model_message is not None
    assert len(model_message["parts"]) >= 1
    
    # Check that the text part has the thoughtSignature
    text_part = model_message["parts"][0]
    assert text_part["text"] == "Hi there"
    assert "thoughtSignature" in text_part
    assert text_part["thoughtSignature"] == "sig-round-trip-abc"


def test_round_trip_without_thought_signature_still_works():
    """
    Test that messages without thoughtSignatures continue to work normally.
    This ensures backward compatibility.
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hi there"
        },
        {"role": "user", "content": "How are you?"}
    ]
    
    gemini_contents = _gemini_convert_messages_with_history(messages)
    
    # Find the assistant (model) message
    model_message = None
    for content in gemini_contents:
        if content.get("role") == "model":
            model_message = content
            break
    
    assert model_message is not None
    assert len(model_message["parts"]) >= 1
    
    # Check that the text part works without thoughtSignature
    text_part = model_message["parts"][0]
    assert text_part["text"] == "Hi there"
    assert "thoughtSignature" not in text_part
