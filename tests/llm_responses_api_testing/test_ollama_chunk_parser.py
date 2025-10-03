import pytest
from unittest.mock import patch
from litellm.llms.ollama.chat.transformation import OllamaChatCompletionResponseIterator
from litellm.types.utils import Delta, StreamingChoices

# Mock the Delta and other necessary classes if they are not directly importable or need special setup
class MockModelResponseStream:
    def __init__(self, choices, model, object_type, system_fingerprint, usage=None, **kwargs):
        self.choices = choices
        self.model = model
        self.object = object_type
        self.system_fingerprint = system_fingerprint
        self.usage = usage
        for key, value in kwargs.items():
            setattr(self, key, value)

@pytest.fixture
def mock_iterator():
    """Fixture to create a mock OllamaChatCompletionResponseIterator."""
    iterator = OllamaChatCompletionResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        json_mode=False
    )
    return iterator

def test_full_think_block_in_one_chunk(mock_iterator):
    """Test case where a complete <think>...</think> block is in a single chunk."""
    chunk = {"message": {"content": "<think>This is a thought.</think>"}, "done": False, "model": "test-model"}
    with patch("litellm.llms.ollama.chat.transformation.uuid.uuid4", return_value="1234"):
        result = mock_iterator.chunk_parser(chunk)
    assert result.choices[0].delta.content == ""
    assert result.choices[0].delta.reasoning_content == "This is a thought."
    assert mock_iterator.started_reasoning_content
    assert mock_iterator.finished_reasoning_content

def test_think_tags_split_across_chunks(mock_iterator):
    """Test case where <think> and </think> tags are in separate chunks."""
    chunk1 = {"message": {"content": "<think>This is a thought."}, "done": False, "model": "test-model"}
    chunk2 = {"message": {"content": " And it continues.</think>"}, "done": True, "model": "test-model"}

    with patch("litellm.llms.ollama.chat.transformation.uuid.uuid4", return_value="1234"):
        result1 = mock_iterator.chunk_parser(chunk1)
        assert result1.choices[0].delta.reasoning_content == "This is a thought."
        assert mock_iterator.started_reasoning_content
        assert not mock_iterator.finished_reasoning_content

        result2 = mock_iterator.chunk_parser(chunk2)
        assert result2.choices[0].delta.reasoning_content == " And it continues."
        assert mock_iterator.started_reasoning_content
        assert mock_iterator.finished_reasoning_content

def test_content_before_and_after_think_tag(mock_iterator):
    """Test case where there is content before and after the <think> ... </think> block"""
    chunk = {"message": {"content": "Here is a preamble. <think>This is a thought.</think> Here is a postamble."}, "done": True, "model": "test-model"}
    
    with patch("litellm.llms.ollama.chat.transformation.uuid.uuid4", return_value="1234"):
        result = mock_iterator.chunk_parser(chunk)

    assert result.choices[0].delta.content == "Here is a preamble. Here is a postamble."
    assert result.choices[0].delta.reasoning_content == "This is a thought."
    assert mock_iterator.started_reasoning_content
    assert mock_iterator.finished_reasoning_content

@patch('litellm.llms.ollama.chat.transformation.OllamaChatCompletionResponseIterator.construct_empty_chunk', create=True)
def test_whitespace_chunks(mock_construct_empty_chunk, mock_iterator):
    """Test case where chunks contain only whitespace."""
    mock_construct_empty_chunk.return_value = MockModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content="", reasoning_content=None, role="assistant", tool_calls=None), finish_reason=None)],
            model="test-model",
            object_type="chat.completion.chunk",
            system_fingerprint=None
    )
    chunk1 = {"message": {"content": " "}, "done": False, "model": "test-model"}
    chunk2 = {"message": {"content": "\n\n"}, "done": True, "model": "test-model"}

    result1 = mock_iterator.chunk_parser(chunk1)
    assert result1.choices[0].delta.content == " "
    assert result1.choices[0].delta.reasoning_content == ""

    result2 = mock_iterator.chunk_parser(chunk2)
    assert result2.choices[0].delta.content == "\n\n"
    assert result2.choices[0].delta.reasoning_content == ""

def test_content_before_think_tag(mock_iterator):
    """Test case where there is regular content before the <think> tag in the same chunk."""
    chunk = {"message": {"content": "Regular content <think>starting thought"}, "done": False, "model": "test-model"}
    
    with patch("litellm.llms.ollama.chat.transformation.uuid.uuid4", return_value="1234"):
        result = mock_iterator.chunk_parser(chunk)
    
    assert result.choices[0].delta.content == "Regular content "
    assert result.choices[0].delta.reasoning_content == "starting thought"
    assert mock_iterator.started_reasoning_content
    assert not mock_iterator.finished_reasoning_content

def test_content_after_think_end_tag(mock_iterator):
    """Test case where there is regular content after the </think> tag in the same chunk."""
    # First start the reasoning
    chunk1 = {"message": {"content": "<think>This is a thought"}, "done": False, "model": "test-model"}
    with patch("litellm.llms.ollama.chat.transformation.uuid.uuid4", return_value="1234"):
        mock_iterator.chunk_parser(chunk1)
    
    # Then end it with content after
    chunk2 = {"message": {"content": " continued.</think> More regular content"}, "done": True, "model": "test-model"}
    with patch("litellm.llms.ollama.chat.transformation.uuid.uuid4", return_value="1234"):
        result = mock_iterator.chunk_parser(chunk2)
    
    assert result.choices[0].delta.reasoning_content == " continued."
    assert result.choices[0].delta.content == " More regular content"
    assert mock_iterator.started_reasoning_content
    assert mock_iterator.finished_reasoning_content

def test_mixed_content_across_multiple_chunks(mock_iterator):
    """Test case with mixed content and reasoning across multiple chunks."""
    chunk1 = {"message": {"content": "Hello "}, "done": False, "model": "test-model"}
    chunk2 = {"message": {"content": "world <think>I'm thinking"}, "done": False, "model": "test-model"}
    chunk3 = {"message": {"content": " about this</think> and "}, "done": False, "model": "test-model"}
    chunk4 = {"message": {"content": "continuing."}, "done": True, "model": "test-model"}
    
    with patch("litellm.llms.ollama.chat.transformation.uuid.uuid4", return_value="1234"):
        # Chunk 1: Regular content before any reasoning
        result1 = mock_iterator.chunk_parser(chunk1)
        assert result1.choices[0].delta.content == "Hello "
        assert result1.choices[0].delta.reasoning_content == ""
        assert not mock_iterator.started_reasoning_content
        
        # Chunk 2: Content before <think> and start of reasoning
        result2 = mock_iterator.chunk_parser(chunk2)
        assert result2.choices[0].delta.content == "world "
        assert result2.choices[0].delta.reasoning_content == "I'm thinking"
        assert mock_iterator.started_reasoning_content
        assert not mock_iterator.finished_reasoning_content
        
        # Chunk 3: End of reasoning and content after </think>
        result3 = mock_iterator.chunk_parser(chunk3)
        assert result3.choices[0].delta.reasoning_content == " about this"
        assert result3.choices[0].delta.content == " and "
        assert mock_iterator.finished_reasoning_content
        
        # Chunk 4: Regular content after reasoning finished
        result4 = mock_iterator.chunk_parser(chunk4)
        assert result4.choices[0].delta.content == "continuing."
        assert result4.choices[0].delta.reasoning_content == ""
