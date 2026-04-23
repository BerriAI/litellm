"""
Unit tests for litellm.compress().
"""

import os
import importlib

import pytest

import litellm
from litellm.compression.scoring.bm25 import bm25_score_messages
from litellm.compression.scoring.embedding_scorer import embedding_score_messages
from litellm.compression.content_detection import detect_content_type
from litellm.compression.message_stubbing import extract_key, stub_message
from litellm.compression.retrieval_tool import build_retrieval_tool
from litellm.types.utils import CallTypes

CALL_TYPE = CallTypes.completion
ANTHROPIC_CALL_TYPE = CallTypes.anthropic_messages


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
    result = litellm.compress(messages, model="gpt-4o", call_type=CALL_TYPE)
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
        call_type=CALL_TYPE,
        compression_trigger=1000,
        compression_target=500,
    )

    assert result["compressed_tokens"] < result["original_tokens"]
    assert result["compression_ratio"] > 0
    assert len(result["cache"]) > 0
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "litellm_content_retrieve"


def test_compress_anthropic_list_content_is_boundary_stable():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "System prompt"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "# a.py\n" + "alpha " * 2000},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/a.png"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "# b.py\n" + "beta " * 2000},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/b.png"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Fix alpha bug in a.py"}],
        },
    ]

    result = litellm.compress(
        messages=messages,
        model="claude-sonnet-4-20250514",
        call_type=ANTHROPIC_CALL_TYPE,
        compression_trigger=1000,
        compression_target=500,
    )

    assert result["compressed_tokens"] < result["original_tokens"]
    assert len(result["messages"]) == len(messages)
    assert [m["role"] for m in result["messages"]] == [m["role"] for m in messages]
    assert len(result["cache"]) > 0
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "custom"
    assert result["tools"][0]["name"] == "litellm_content_retrieve"
    assert "input_schema" in result["tools"][0]


def test_compress_preserves_system_message():
    messages = [
        {"role": "system", "content": "System prompt. " * 500},
        {"role": "user", "content": "Large file content. " * 5000},
        {"role": "user", "content": "Fix the bug"},
    ]
    result = litellm.compress(
        messages, model="gpt-4o", call_type=CALL_TYPE, compression_trigger=1000
    )
    assert result["messages"][0]["role"] == "system"
    assert "System prompt" in result["messages"][0]["content"]


def test_compress_preserves_last_user_message():
    messages = [
        {"role": "user", "content": "Big context " * 5000},
        {"role": "user", "content": "Fix the bug in auth.py"},
    ]
    result = litellm.compress(
        messages, model="gpt-4o", call_type=CALL_TYPE, compression_trigger=1000
    )
    last_user = [m for m in result["messages"] if m["role"] == "user"][-1]
    assert "Fix the bug in auth.py" in last_user["content"]


def test_compress_preserves_last_assistant_message():
    messages = [
        {"role": "user", "content": "Big context " * 5000},
        {"role": "assistant", "content": "I'll help with that. " * 2000},
        {"role": "user", "content": "Now fix the bug"},
    ]
    result = litellm.compress(
        messages, model="gpt-4o", call_type=CALL_TYPE, compression_trigger=1000
    )
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
    result = litellm.compress(
        messages, model="gpt-4o", call_type=CALL_TYPE, compression_trigger=1000
    )
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
    result = litellm.compress(
        messages, model="gpt-4o", call_type=CALL_TYPE, compression_trigger=2000
    )
    # Should have compressed — target = 1000
    assert result["compressed_tokens"] <= result["original_tokens"]


def test_compress_nested_tool_result_extracts_text_only():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "System rules"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "prefix"},
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "nested text fragment"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/secret-tool.png",
                            },
                        },
                    ],
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/top.png"},
                },
                {"type": "text", "text": " " + ("irrelevant " * 3000)},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "final query that must remain"}],
        },
    ]

    result = litellm.compress(
        messages=messages,
        model="claude-sonnet-4-20250514",
        call_type=ANTHROPIC_CALL_TYPE,
        compression_trigger=500,
        compression_target=100,
    )

    cached_text = " ".join(result["cache"].values())
    assert "nested text fragment" in cached_text
    assert "https://example.com/secret-tool.png" not in cached_text
    assert "https://example.com/top.png" not in cached_text


def test_compress_default_call_type_is_completion():
    result = litellm.compress(
        messages=[
            {"role": "user", "content": "Large context " * 4000},
            {"role": "user", "content": "query"},
        ],
        model="gpt-4o",
        compression_trigger=1000,
        compression_target=500,
    )

    assert result["compressed_tokens"] <= result["original_tokens"]
    assert isinstance(result["tools"], list)


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
        call_type=CALL_TYPE,
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
        call_type=CALL_TYPE,
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
    result = litellm.compress(
        messages, model="gpt-4o", call_type=CALL_TYPE, compression_trigger=1000
    )
    if expected_content == "Unrelated cooking recipes ":
        assert "Unrelated cooking recipes " in result["messages"][1]["content"]
        assert "Authentication code " not in result["messages"][0]["content"]
    elif expected_content == "Authentication code ":
        assert "Authentication code " in result["messages"][0]["content"]
        assert "Unrelated cooking recipes " not in result["messages"][1]["content"]
    else:
        raise ValueError(f"Unexpected expected_content: {expected_content}")


def test_compress_anthropic_drops_irrelevant_tool_exchange_span(monkeypatch):
    compress_module = importlib.import_module("litellm.compression.compress")

    def fake_bm25_score_messages(query, messages):
        assert "final query" in query
        assert len(messages) == 5
        # Prefer idx=0 and de-prioritize the tool exchange span (idx=1,2)
        return [0.95, 0.01, 0.02, 0.8, 1.0]

    def fake_token_counter(model, messages=None, text=None):
        if messages is not None:
            return 1000
        if text is None:
            return 0
        if "final query" in text:
            return 50
        if "assistant_tail" in text:
            return 20
        if "other_blob" in text:
            return 220
        if "tool_payload_relevant" in text:
            return 200
        if text == "":
            return 1
        return 10

    monkeypatch.setattr(
        compress_module, "bm25_score_messages", fake_bm25_score_messages
    )
    monkeypatch.setattr(compress_module, "token_counter", fake_token_counter)

    messages = [
        {"role": "user", "content": "other_blob " * 300},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_drop",
                    "name": "litellm_content_retrieve",
                    "input": {"key": "message_1"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_drop",
                    "content": [{"type": "text", "text": "tool_payload_relevant"}],
                }
            ],
        },
        {"role": "assistant", "content": "assistant_tail"},
        {"role": "user", "content": "final query"},
    ]

    result = litellm.compress(
        messages=messages,
        model="claude-sonnet-4-20250514",
        call_type=ANTHROPIC_CALL_TYPE,
        compression_trigger=100,
        compression_target=280,
    )

    # idx=1,2 should be dropped atomically (no orphan tool blocks left behind)
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "user"
    assert "other_blob" in result["messages"][0]["content"]
    assert result["messages"][1]["content"] == "assistant_tail"
    assert result["messages"][2]["content"] == "final query"
    assert result["cache"] == {}


def test_compress_anthropic_keeps_relevant_tool_exchange_span(monkeypatch):
    compress_module = importlib.import_module("litellm.compression.compress")

    def fake_bm25_score_messages(query, messages):
        assert "final query" in query
        assert len(messages) == 5
        # Prefer the tool exchange span over idx=0
        return [0.05, 0.01, 0.92, 0.8, 1.0]

    def fake_token_counter(model, messages=None, text=None):
        if messages is not None:
            return 1000
        if text is None:
            return 0
        if "final query" in text:
            return 50
        if "assistant_tail" in text:
            return 20
        if "other_blob" in text:
            return 220
        if "tool_payload_relevant" in text:
            return 200
        if text == "":
            return 1
        return 10

    monkeypatch.setattr(
        compress_module, "bm25_score_messages", fake_bm25_score_messages
    )
    monkeypatch.setattr(compress_module, "token_counter", fake_token_counter)

    messages = [
        {"role": "user", "content": "other_blob " * 300},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_keep",
                    "name": "litellm_content_retrieve",
                    "input": {"key": "message_1"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_keep",
                    "content": [{"type": "text", "text": "tool_payload_relevant"}],
                }
            ],
        },
        {"role": "assistant", "content": "assistant_tail"},
        {"role": "user", "content": "final query"},
    ]

    result = litellm.compress(
        messages=messages,
        model="claude-sonnet-4-20250514",
        call_type=ANTHROPIC_CALL_TYPE,
        compression_trigger=100,
        compression_target=280,
    )

    assert len(result["messages"]) == 5
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][2]["role"] == "user"
    # idx=0 should be compressed instead
    assert "litellm_content_retrieve" in result["messages"][0]["content"]
    assert len(result["cache"]) == 1


def test_compress_anthropic_malformed_tool_sequence_passes_through():
    messages = [
        {"role": "user", "content": "other_blob " * 300},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_broken",
                    "name": "litellm_content_retrieve",
                    "input": {"key": "message_1"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "missing tool_result"}]},
        {"role": "user", "content": "final query"},
    ]

    result = litellm.compress(
        messages=messages,
        model="claude-sonnet-4-20250514",
        call_type=ANTHROPIC_CALL_TYPE,
        compression_trigger=100,
        compression_target=280,
    )

    assert result["messages"] == messages
    assert result["cache"] == {}
    assert result["tools"] == []
    assert result["compression_skipped_reason"] == "invalid_anthropic_tool_sequence"
