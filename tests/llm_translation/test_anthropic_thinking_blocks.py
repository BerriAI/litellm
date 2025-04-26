"""
Test for checking Anthropic thinking blocks in streaming responses
"""

import asyncio
import pytest
import litellm

MODEL = "anthropic/claude-3-7-sonnet-20250219"
MAGIC_STRING = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"


@pytest.mark.asyncio
async def test_has_chunks_with_thinking_blocks_regular_prompt():
    """Test that regular prompts have thinking blocks in streaming chunks"""
    # Test with a regular prompt
    block_exists, block_type = await has_chunks_with_thinking_blocks("How are you?")
    assert block_exists, "Regular prompts should have thinking blocks in stream chunks"
    assert block_type == "thinking", "Regular prompts should have thinking blocks in stream chunks"


@pytest.mark.asyncio
async def test_has_chunks_with_thinking_blocks_magic_string():
    """Test that the magic string prompt has thinking blocks in streaming chunks"""
    # Test with the magic string that should also have thinking blocks
    block_exists, block_type = await has_chunks_with_thinking_blocks(MAGIC_STRING)
    assert block_exists, "Magic string prompt should have thinking blocks in stream chunks"
    assert block_type == "redacted_thinking", "Magic string prompt should have redacted thinking blocks in stream chunks"

async def has_chunks_with_thinking_blocks(query: str):
    """
    Check if the streaming response from Claude contains thinking blocks
    
    Args:
        query: The query to send to Claude
        
    Returns:
        bool: True if thinking blocks were found in the streaming response
    """
    stream = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": query}],
        stream=True,
        reasoning_effort="low",
    )

    has_thinking_blocks = False

    async for chunk in stream:
        chunk_delta = chunk.choices[0].delta                        
        if hasattr(chunk_delta, "thinking_blocks") and chunk_delta.thinking_blocks:
            return True, chunk_delta.thinking_blocks[0]["type"]

    return False, None

if __name__ == "__main__":
    asyncio.run(test_has_chunks_with_thinking_blocks_regular_prompt())
    asyncio.run(test_has_chunks_with_thinking_blocks_magic_string())