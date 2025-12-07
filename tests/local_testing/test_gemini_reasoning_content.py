from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig


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
