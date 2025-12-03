import json
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig


def test_empty_part_does_not_create_thinking_block():
    parts = [{"text": "", "thoughtSignature": "sig-1"}]
    config = VertexGeminiConfig()
    thinking_blocks = config._extract_thinking_blocks_from_parts(parts)
    assert thinking_blocks == []


def test_non_empty_part_creates_thinking_block():
    parts = [{"text": "Some thinking", "thoughtSignature": "sig-2"}]
    config = VertexGeminiConfig()
    thinking_blocks = config._extract_thinking_blocks_from_parts(parts)
    assert len(thinking_blocks) == 1
    block = thinking_blocks[0]
    # thinking should be valid JSON containing the text
    parsed = json.loads(block["thinking"]) if isinstance(block["thinking"], str) else None
    assert parsed is not None and parsed.get("text") == "Some thinking"
