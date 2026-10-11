"""
Unit tests for litellm.compress().
"""

import os

import pytest

import litellm
from litellm.types.utils import CallTypes

CALL_TYPE = CallTypes.completion


# ---------------------------------------------------------------------------
# BM25 scorer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Content detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Message stubbing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# compress() — end-to-end
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Embedding scorer — integration test (skipped without API key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Needs OPENAI_API_KEY")
def test_embedding_scorer():
    result = litellm.compress(
        messages=[
            {"role": "user", "content": "Authentication code " * 2000},
            {"role": "user", "content": "Unrelated cooking recipes " * 2000},
            {"role": "user", "content": "Fix auth"},
        ],
        model="gpt-4o",
        call_type=CALL_TYPE,
        compression_trigger=1000,
        embedding_model="text-embedding-3-small",
    )
    assert result["compression_ratio"] > 0
    assert len(result["cache"]) > 0
