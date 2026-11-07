from typing import Final

import pytest

from litellm.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from litellm.rag.text_splitters.recursive_character_text_splitter import (
    RecursiveCharacterTextSplitter,
)


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [
        (100, 100),
        (100, 150),
        (1, 1),
    ],
)
def test_chunk_overlap_at_least_chunk_size_is_rejected(
    chunk_size: int, chunk_overlap: int
) -> None:
    with pytest.raises(
        ValueError, match="chunk_overlap must be smaller than chunk_size"
    ):
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )


@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_non_positive_chunk_size_is_rejected(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveCharacterTextSplitter(chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_overlap", [-1, -100])
def test_negative_chunk_overlap_is_rejected(chunk_overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RecursiveCharacterTextSplitter(chunk_overlap=chunk_overlap)


def test_force_split_path_terminates_with_valid_overlap() -> None:
    splitter: Final = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=2, separators=["\n\n"]
    )
    chunks: Final = splitter.split_text("abcdefghijklmnop")
    assert chunks == ["abcdefghij", "ijklmnop"]


def test_valid_overlap_keeps_splitting() -> None:
    splitter: Final = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
    chunks: Final = splitter.split_text("abcdefghijklmnop")
    assert chunks == ["abcdefghij", "ijklmnop"]


def test_defaults_construct_without_error() -> None:
    splitter: Final = RecursiveCharacterTextSplitter()
    assert splitter.chunk_size == DEFAULT_CHUNK_SIZE
    assert splitter.chunk_overlap == DEFAULT_CHUNK_OVERLAP
