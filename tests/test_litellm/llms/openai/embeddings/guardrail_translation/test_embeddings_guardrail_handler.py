"""
Test OpenAI Embeddings Guardrail Translation Handler
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.openai.embeddings.guardrail_translation.handler import (
    OpenAIEmbeddingsHandler,
)
from litellm.types.utils import CallTypes


@pytest.mark.asyncio
async def test_embeddings_handler_string_input():
    """Test embeddings handler with single string input"""
    handler = OpenAIEmbeddingsHandler()
    
    # Mock guardrail
    mock_guardrail = MagicMock()
    mock_guardrail.apply_guardrail = AsyncMock(return_value={"texts": ["processed text"]})
    
    data = {
        "input": "Hello, world!",
        "model": "text-embedding-3-small"
    }
    
    result = await handler.process_input_messages(
        data=data,
        guardrail_to_apply=mock_guardrail,
    )
    
    # Verify guardrail was called with correct inputs
    mock_guardrail.apply_guardrail.assert_called_once()
    call_args = mock_guardrail.apply_guardrail.call_args
    assert call_args.kwargs["inputs"]["texts"] == ["Hello, world!"]
    assert call_args.kwargs["inputs"]["model"] == "text-embedding-3-small"
    
    # Verify result
    assert result["input"] == "processed text"


@pytest.mark.asyncio
async def test_embeddings_handler_list_of_strings_input():
    """Test embeddings handler with list of strings input"""
    handler = OpenAIEmbeddingsHandler()
    
    # Mock guardrail
    mock_guardrail = MagicMock()
    mock_guardrail.apply_guardrail = AsyncMock(
        return_value={"texts": ["processed text 1", "processed text 2"]}
    )
    
    data = {
        "input": ["Hello, world!", "How are you?"],
        "model": "text-embedding-3-small"
    }
    
    result = await handler.process_input_messages(
        data=data,
        guardrail_to_apply=mock_guardrail,
    )
    
    # Verify guardrail was called with correct inputs
    mock_guardrail.apply_guardrail.assert_called_once()
    call_args = mock_guardrail.apply_guardrail.call_args
    assert call_args.kwargs["inputs"]["texts"] == ["Hello, world!", "How are you?"]
    
    # Verify result
    assert result["input"] == ["processed text 1", "processed text 2"]


def test_embeddings_guardrail_translation_mappings():
    """Test that embeddings handler is registered for correct call types"""
    from litellm.llms.openai.embeddings.guardrail_translation import (
        guardrail_translation_mappings,
    )
    
    assert CallTypes.embedding in guardrail_translation_mappings
    assert CallTypes.aembedding in guardrail_translation_mappings
    assert guardrail_translation_mappings[CallTypes.embedding] == OpenAIEmbeddingsHandler
    assert guardrail_translation_mappings[CallTypes.aembedding] == OpenAIEmbeddingsHandler
