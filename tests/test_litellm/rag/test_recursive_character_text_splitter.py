import pytest
from litellm.rag.text_splitters.recursive_character_text_splitter import (
    RecursiveCharacterTextSplitter,
)


def test_invalid_chunk_overlap_greater_than_or_equal_chunk_size():
    """chunk_overlap >= chunk_size must raise ValueError to prevent infinite loops."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=1000)


def test_invalid_chunk_size_non_positive():
    """chunk_size <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="chunk_size"):
        RecursiveCharacterTextSplitter(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size"):
        RecursiveCharacterTextSplitter(chunk_size=-10, chunk_overlap=0)


def test_invalid_chunk_overlap_negative():
    """chunk_overlap < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=-1)


def test_split_text_basic():
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    text = "This is a test paragraph.\n\nThis is another paragraph that is longer and will need splitting."
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50


def test_force_split_no_matching_separators():
    """Verify that text with no matching separators splits cleanly without hanging."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5, separators=["\n\n"])
    text = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert "".join(chunks) != ""
