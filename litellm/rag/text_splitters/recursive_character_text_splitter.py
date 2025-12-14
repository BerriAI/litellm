"""
RecursiveCharacterTextSplitter for RAG ingestion.

A simple implementation that splits text recursively by different separators.
"""

from typing import List, Optional

from litellm.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


class RecursiveCharacterTextSplitter:
    """
    Split text recursively by different separators.

    Tries to split by the first separator, then recursively splits
    by subsequent separators if chunks are still too large.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str], depth: int = 0) -> List[str]:
        """Recursively split text using separators."""
        from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

        if depth > DEFAULT_MAX_RECURSE_DEPTH:
            # Max depth reached, return text as-is split into chunk_size pieces
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        final_chunks: List[str] = []

        # Get the appropriate separator
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split by the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks
        good_splits: List[str] = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Chunk is too big, merge what we have and recurse
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if new_separators:
                    # Recursively split with finer separators
                    other_chunks = self._split_text(split, new_separators, depth + 1)
                    final_chunks.extend(other_chunks)
                else:
                    # No more separators, force split
                    final_chunks.extend(self._force_split(split))

        # Merge remaining good splits
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks respecting chunk_size and chunk_overlap."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)
            sep_len = len(separator) if current_chunk else 0

            if current_length + split_len + sep_len > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Handle overlap
                    while current_length > self.chunk_overlap and len(current_chunk) > 1:
                        removed = current_chunk.pop(0)
                        current_length -= len(removed) + len(separator)

            current_chunk.append(split)
            current_length += split_len + sep_len

        # Add remaining
        if current_chunk:
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """Force split text by chunk_size when no separator works."""
        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap if end < len(text) else len(text)

        return chunks

