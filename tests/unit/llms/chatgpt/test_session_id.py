"""Session-id resolution for the ChatGPT provider.

ChatGPT partitions the Codex prompt cache by the Responses session-id
header. These tests pin the resolution order:
explicit operator config > prompt_cache_key > litellm-internal ids,
with proxy-generated session ids skipped entirely.
"""

from litellm.llms.chatgpt.common_utils import (
    ensure_chatgpt_session_id,
    get_chatgpt_session_id,
)


def test_explicit_session_ids_win():
    assert get_chatgpt_session_id({"session_id": "s", "prompt_cache_key": "k"}) == "s"
    assert get_chatgpt_session_id({"litellm_session_id": "ls", "prompt_cache_key": "k"}) == "ls"
    assert get_chatgpt_session_id({"metadata": {"session_id": "ms"}, "prompt_cache_key": "k"}) == "ms"


def test_prompt_cache_key_becomes_session_id():
    assert get_chatgpt_session_id({"prompt_cache_key": "conv-abc"}) == "conv-abc"
    assert ensure_chatgpt_session_id({"prompt_cache_key": "conv-abc"}) == "conv-abc"


def test_prompt_cache_key_beats_internal_request_ids():
    # litellm_trace_id / litellm_call_id are per-request; they must not
    # shadow a stable per-conversation key.
    assert get_chatgpt_session_id({"litellm_call_id": "c", "prompt_cache_key": "k"}) == "k"
    assert get_chatgpt_session_id({"litellm_trace_id": "t", "prompt_cache_key": "k"}) == "k"


def test_prompt_cache_key_is_sanitized_for_header_use():
    # Caller-controlled value headed for an HTTP header: control chars are
    # replaced (h11 field-value validation would otherwise 500 the request),
    # keeping the key stable.
    assert get_chatgpt_session_id({"prompt_cache_key": "a\nb"}) == "a_b"


def test_generated_session_ids_are_skipped():
    # missing_session_id: "generate" stamps a per-request id in every channel;
    # it must not be used as identity, and the cache key must still win.
    for metadata_key in ("metadata", "litellm_metadata"):
        generated = {
            "litellm_session_id": "gen",
            metadata_key: {"session_id": "gen", "litellm_session_id_generated": True},
        }
        assert get_chatgpt_session_id({**generated, "prompt_cache_key": "k"}) == "k"
        assert get_chatgpt_session_id(generated) is None
        a, b = ensure_chatgpt_session_id(generated), ensure_chatgpt_session_id(generated)
        assert a != "gen" and b != "gen" and a != b


def test_uuid4_fallback_without_any_key():
    a, b = ensure_chatgpt_session_id({}), ensure_chatgpt_session_id({})
    assert a and b and a != b
