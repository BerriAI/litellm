"""
Unit tests for litellm.compress().
"""

import os

import pytest

import litellm
from litellm.compression.scoring.bm25 import bm25_score_messages
from litellm.compression.scoring.embedding_scorer import embedding_score_messages
from litellm.compression.content_detection import detect_content_type
from litellm.compression.message_stubbing import extract_key, stub_message
from litellm.compression.retrieval_tool import build_retrieval_tool


# ---------------------------------------------------------------------------
# BM25 scorer
# ---------------------------------------------------------------------------


def test_bm25_relevance_ranking():
    query = "Fix the authentication bug in the login handler"
    messages = [
        {
            "role": "user",
            "content": "def login_handler(): authentication check bug fix",
        },
        {"role": "user", "content": "def render_template(name): css styling layout"},
        {"role": "user", "content": "def verify(): authentication token bug handler"},
    ]
    scores = bm25_score_messages(query, messages)
    # Messages sharing query terms should score higher than unrelated ones
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_bm25_empty_query():
    scores = bm25_score_messages("", [{"role": "user", "content": "hello"}])
    assert scores == [0.0]


def test_bm25_empty_messages():
    scores = bm25_score_messages("query", [])
    assert scores == []


def test_bm25_empty_content():
    scores = bm25_score_messages("query", [{"role": "user", "content": ""}])
    assert scores == [0.0]


# ---------------------------------------------------------------------------
# Content detection
# ---------------------------------------------------------------------------


def test_detect_code():
    code = """
import os
from pathlib import Path

def main():
    class Foo:
        pass
    return Foo()
"""
    assert detect_content_type(code) == "code"


def test_detect_json():
    assert detect_content_type('{"key": "value", "num": 42}') == "json"
    assert detect_content_type("[1, 2, 3]") == "json"


def test_detect_text():
    assert detect_content_type("This is a plain text paragraph about dogs.") == "text"


def test_detect_empty():
    assert detect_content_type("") == "text"


# ---------------------------------------------------------------------------
# Message stubbing
# ---------------------------------------------------------------------------


def test_extract_key_with_filename():
    msg = {"role": "user", "content": "# auth.py\ndef authenticate():\n    pass"}
    used: set = set()
    key = extract_key(msg, fallback_index=0, used_keys=used)
    assert key == "auth.py"


def test_extract_key_fallback():
    msg = {"role": "user", "content": "Some random content without a filename"}
    used: set = set()
    key = extract_key(msg, fallback_index=5, used_keys=used)
    assert key == "message_5"


def test_extract_key_duplicates():
    used: set = set()
    msg = {"role": "user", "content": "# auth.py\ncode here"}
    k1 = extract_key(msg, fallback_index=0, used_keys=used)
    k2 = extract_key(msg, fallback_index=1, used_keys=used)
    assert k1 == "auth.py"
    assert k2 == "auth.py_2"


def test_stub_message():
    msg = {"role": "user", "content": "line1\nline2\nline3"}
    stubbed = stub_message(msg, "test_key")
    assert stubbed["role"] == "user"
    assert "test_key" in stubbed["content"]
    assert "litellm_content_retrieve" in stubbed["content"]
    assert "3 lines" in stubbed["content"]


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------


def test_retrieval_tool_schema():
    tool = build_retrieval_tool(["auth.py", "utils.py"])
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "litellm_content_retrieve"
    assert "key" in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["properties"]["key"]["enum"] == [
        "auth.py",
        "utils.py",
    ]
    assert tool["function"]["parameters"]["required"] == ["key"]


def test_retrieval_tool_description_lists_keys():
    tool = build_retrieval_tool(["foo.py", "bar.js"])
    desc = tool["function"]["description"]
    assert "foo.py" in desc
    assert "bar.js" in desc


# ---------------------------------------------------------------------------
# compress() — end-to-end
# ---------------------------------------------------------------------------


def test_compress_below_trigger_passthrough():
    messages = [{"role": "user", "content": "hello"}]
    result = litellm.compress(messages, model="gpt-4o")
    assert result["messages"] == messages
    assert result["cache"] == {}
    assert result["tools"] == []
    assert result["compression_ratio"] == 0.0
    assert result["original_tokens"] == result["compressed_tokens"]


def test_compress_above_trigger():
    big_messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {
            "role": "user",
            "content": "# auth.py\n" + "def authenticate():\n    pass\n" * 2000,
        },
        {
            "role": "user",
            "content": "# utils.py\n" + "def helper():\n    pass\n" * 2000,
        },
        {
            "role": "user",
            "content": "# readme.md\n" + "This is documentation. " * 2000,
        },
        {"role": "user", "content": "Fix the bug in auth.py"},
    ]

    result = litellm.compress(
        big_messages,
        model="gpt-4o",
        compression_trigger=1000,
        compression_target=500,
    )

    assert result["compressed_tokens"] < result["original_tokens"]
    assert result["compression_ratio"] > 0
    assert len(result["cache"]) > 0
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "litellm_content_retrieve"


def test_compress_preserves_system_message():
    messages = [
        {"role": "system", "content": "System prompt. " * 500},
        {"role": "user", "content": "Large file content. " * 5000},
        {"role": "user", "content": "Fix the bug"},
    ]
    result = litellm.compress(messages, model="gpt-4o", compression_trigger=1000)
    assert result["messages"][0]["role"] == "system"
    assert "System prompt" in result["messages"][0]["content"]


def test_compress_preserves_last_user_message():
    messages = [
        {"role": "user", "content": "Big context " * 5000},
        {"role": "user", "content": "Fix the bug in auth.py"},
    ]
    result = litellm.compress(messages, model="gpt-4o", compression_trigger=1000)
    last_user = [m for m in result["messages"] if m["role"] == "user"][-1]
    assert "Fix the bug in auth.py" in last_user["content"]


def test_compress_preserves_last_assistant_message():
    messages = [
        {"role": "user", "content": "Big context " * 5000},
        {"role": "assistant", "content": "I'll help with that. " * 2000},
        {"role": "user", "content": "Now fix the bug"},
    ]
    result = litellm.compress(messages, model="gpt-4o", compression_trigger=1000)
    assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    # The last assistant message should be preserved (not stubbed)
    last_assistant = assistant_msgs[-1]
    assert "I'll help with that" in last_assistant["content"]


def test_cache_keys_match_stubs():
    messages = [
        {"role": "user", "content": "# auth.py\n" + "code " * 5000},
        {"role": "user", "content": "Fix it"},
    ]
    result = litellm.compress(messages, model="gpt-4o", compression_trigger=1000)
    if result["tools"]:
        tool_desc = result["tools"][0]["function"]["description"]
        for key in result["cache"]:
            assert key in tool_desc


def test_compress_default_target():
    """compression_target defaults to compression_trigger // 2."""
    messages = [
        {"role": "user", "content": "content " * 5000},
        {"role": "user", "content": "query"},
    ]
    result = litellm.compress(messages, model="gpt-4o", compression_trigger=2000)
    # Should have compressed — target = 1000
    assert result["compressed_tokens"] <= result["original_tokens"]


def test_compress_forwards_embedding_model_params(monkeypatch):
    captured = {}

    def fake_embedding_score_messages(
        query, messages, model, cache=None, embedding_model_params=None
    ):
        captured["query"] = query
        captured["model"] = model
        captured["embedding_model_params"] = embedding_model_params
        return [0.0] * len(messages)

    monkeypatch.setattr(
        "litellm.compression.scoring.embedding_scorer.embedding_score_messages",
        fake_embedding_score_messages,
    )

    result = litellm.compress(
        messages=[
            {"role": "user", "content": "Authentication code " * 2000},
            {"role": "user", "content": "Fix auth"},
        ],
        model="gpt-4o",
        compression_trigger=1000,
        embedding_model="text-embedding-3-small",
        embedding_model_params={"api_base": "https://example-embeddings.test"},
    )

    assert result["compressed_tokens"] <= result["original_tokens"]
    assert captured["model"] == "text-embedding-3-small"
    assert captured["embedding_model_params"] == {
        "api_base": "https://example-embeddings.test"
    }


def test_embedding_scorer_forwards_embedding_model_params(monkeypatch):
    captured = {}

    class _MockResponse:
        data = [
            {"embedding": [1.0, 0.0]},
            {"embedding": [1.0, 0.0]},
            {"embedding": [0.0, 1.0]},
        ]

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return _MockResponse()

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    scores = embedding_score_messages(
        query="auth",
        messages=[
            {"role": "user", "content": "auth code"},
            {"role": "user", "content": "cooking recipe"},
        ],
        model="text-embedding-3-small",
        embedding_model_params={"api_base": "https://example-embeddings.test"},
    )

    assert len(scores) == 2
    assert captured["model"] == "text-embedding-3-small"
    assert captured["api_base"] == "https://example-embeddings.test"


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
        compression_trigger=1000,
        embedding_model="text-embedding-3-small",
    )
    assert result["compression_ratio"] > 0
    assert len(result["cache"]) > 0


@pytest.mark.parametrize(
    "final_user_message, expected_content",
    [
        ("How to cook?", "Unrelated cooking recipes "),
        ("Fix auth", "Authentication code "),
    ],
)
def test_simple_compression(final_user_message, expected_content):
    messages = [
        {"role": "user", "content": "Authentication code " * 2000},
        {"role": "user", "content": "Unrelated cooking recipes " * 2000},
        {"role": "user", "content": final_user_message},
    ]
    result = litellm.compress(messages, model="gpt-4o", compression_trigger=1000)
    print(result["messages"])
    if expected_content == "Unrelated cooking recipes ":
        assert "Unrelated cooking recipes " in result["messages"][1]["content"]
        assert "Authentication code " not in result["messages"][0]["content"]
    elif expected_content == "Authentication code ":
        assert "Authentication code " in result["messages"][0]["content"]
        assert "Unrelated cooking recipes " not in result["messages"][1]["content"]
    else:
        raise ValueError(f"Unexpected expected_content: {expected_content}")
