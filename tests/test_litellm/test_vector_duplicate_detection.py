"""
Tests for vector-based semantic duplicate detection.

Tests the core logic (index, query, similarity) without requiring
API keys or GitHub access.
"""

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Unit tests for cosine_similarity
# ---------------------------------------------------------------------------

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "vector_duplicate_detection",
    os.path.join(os.path.dirname(__file__), "../../.github/scripts/vector_duplicate_detection.py"),
)
vdd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vdd)


def test_cosine_similarity_identical():
    """Identical vectors have similarity 1.0."""
    vec = [1.0, 2.0, 3.0]
    assert math.isclose(vdd.cosine_similarity(vec, vec), 1.0, abs_tol=1e-9)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors have similarity 0.0."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert math.isclose(vdd.cosine_similarity(a, b), 0.0, abs_tol=1e-9)


def test_cosine_similarity_opposite():
    """Opposite vectors have similarity -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert math.isclose(vdd.cosine_similarity(a, b), -1.0, abs_tol=1e-9)


def test_cosine_similarity_zero_vector():
    """Zero vector returns 0.0."""
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert vdd.cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Unit tests for _item_to_text
# ---------------------------------------------------------------------------


def test_item_to_text_full():
    """Converts item with title, body, and labels to text."""
    item = {
        "title": "Fix auth bug",
        "body": "The API key validation fails when...",
        "labels": [{"name": "bug"}, {"name": "auth"}],
    }
    text = vdd._item_to_text(item)
    assert "Fix auth bug" in text
    assert "API key validation" in text
    assert "bug" in text
    assert "auth" in text


def test_item_to_text_no_body():
    """Handles missing body gracefully."""
    item = {"title": "Simple title", "body": None, "labels": []}
    text = vdd._item_to_text(item)
    assert text == "Simple title"


def test_item_to_text_truncates_long_body():
    """Truncates body longer than 2000 chars."""
    item = {"title": "Title", "body": "x" * 5000, "labels": []}
    text = vdd._item_to_text(item)
    # Title + newline + newline + truncated body
    assert len(text) < 2100


# ---------------------------------------------------------------------------
# Unit tests for VectorIndex
# ---------------------------------------------------------------------------


def test_vector_index_save_and_load():
    """Index can be saved and loaded from disk."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        index = vdd.VectorIndex(path=path)
        index.items = [
            {"type": "issue", "number": 1, "title": "Test", "labels": []},
            {"type": "pr", "number": 2, "title": "Fix", "labels": ["bug"]},
        ]
        index.vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        index.model = "test-model"
        index.save()

        # Load into a new instance
        with patch.object(vdd, "EMBEDDING_MODEL", "test-model"):
            index2 = vdd.VectorIndex(path=path)
            assert index2.load() is True
            assert len(index2.items) == 2
            assert len(index2.vectors) == 2
            assert index2.items[0]["title"] == "Test"
    finally:
        os.unlink(path)


def test_vector_index_model_mismatch():
    """Loading fails if the model doesn't match."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"model": "old-model", "items": [], "vectors": []}, f)
        path = f.name

    try:
        with patch.object(vdd, "EMBEDDING_MODEL", "new-model"):
            index = vdd.VectorIndex(path=path)
            assert index.load() is False
    finally:
        os.unlink(path)


def test_vector_index_add_items_deduplicates():
    """Adding the same item twice doesn't create duplicates."""
    index = vdd.VectorIndex(path="/dev/null")
    index._indexed_keys = {"issue:1"}
    index.items = [{"type": "issue", "number": 1, "title": "Existing", "labels": []}]
    index.vectors = [[0.1, 0.2]]

    items = [{"number": 1, "title": "Existing", "labels": []}]
    with patch.object(vdd, "get_embeddings", return_value=[[0.1, 0.2]]):
        added = index.add_items(items, "issue")

    assert added == 0
    assert len(index.items) == 1


def test_vector_index_query():
    """Query returns items above threshold, excluding self."""
    index = vdd.VectorIndex(path="/dev/null")
    index.items = [
        {"type": "issue", "number": 1, "title": "Auth bug", "labels": []},
        {"type": "issue", "number": 2, "title": "Auth fix", "labels": []},
        {"type": "pr", "number": 3, "title": "Unrelated", "labels": []},
    ]
    # Make items 1 and 2 similar, item 3 different
    index.vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.1, 0.0],  # very similar to item 1
        [0.0, 0.0, 1.0],   # orthogonal
    ]

    # Mock embedding of query to be similar to items 1 and 2
    with patch.object(vdd, "get_embeddings", return_value=[[1.0, 0.05, 0.0]]):
        results = index.query(
            "Auth problem",
            threshold=0.9,
            exclude_type="issue",
            exclude_number=1,
        )

    # Should find item 2 (similar), exclude item 1 (self), skip item 3 (below threshold)
    assert len(results) == 1
    assert results[0][0]["number"] == 2
    assert results[0][1] > 0.9


def test_vector_index_query_empty():
    """Query on empty index returns empty list."""
    index = vdd.VectorIndex(path="/dev/null")
    with patch.object(vdd, "get_embeddings", return_value=[[1.0, 0.0]]):
        results = index.query("anything")
    assert results == []
