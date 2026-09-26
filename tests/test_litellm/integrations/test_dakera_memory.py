"""Tests for the Dakera persistent-memory logger.

Each test injects a mocked async Dakera client into DakeraMemoryLogger (via the
``_client`` attribute) so no real SDK or network is required. The assertions pin
the observable contract: the pre-call hook recalls with the right namespace/query
and injects a memory system message in the right position, and the success hook
persists the full exchange under the session namespace.
"""

import os
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.dakera_memory import DakeraMemoryLogger, _extract_text


def _client_recalling(*contents: str) -> MagicMock:
    client = MagicMock()
    memories = [SimpleNamespace(content=c) for c in contents]
    client.recall = AsyncMock(return_value=SimpleNamespace(memories=memories))
    client.store_memory = AsyncMock(return_value={})
    return client


def _logger_with(client: MagicMock, **kwargs) -> DakeraMemoryLogger:
    logger = DakeraMemoryLogger(**kwargs)
    logger._client = client
    return logger


@pytest.mark.asyncio
async def test_pre_call_injects_recalled_memories_after_system_before_user():
    client = _client_recalling("alice prefers dark mode", "alice is in tokyo")
    logger = _logger_with(client, top_k=3)
    data = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "what are my prefs?"},
        ],
        "metadata": {"session_id": "user-alice"},
    }

    out = await logger.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    client.recall.assert_awaited_once_with(agent_id="user-alice", query="what are my prefs?", top_k=3)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "system", "user"]
    injected = out["messages"][1]["content"]
    assert "alice prefers dark mode" in injected
    assert "alice is in tokyo" in injected
    # original system prompt must survive and stay first
    assert out["messages"][0]["content"] == "you are helpful"


@pytest.mark.asyncio
async def test_pre_call_no_memories_leaves_messages_untouched():
    client = _client_recalling()  # empty recall
    logger = _logger_with(client)
    original = [{"role": "user", "content": "hi"}]
    data = {"messages": list(original), "metadata": {"session_id": "s1"}}

    out = await logger.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert out["messages"] == original


@pytest.mark.asyncio
async def test_pre_call_uses_last_user_message_as_query():
    client = _client_recalling("m")
    logger = _logger_with(client)
    data = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second and latest"},
        ],
        "metadata": {"session_id": "s1"},
    }

    await logger.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert client.recall.await_args.kwargs["query"] == "second and latest"


@pytest.mark.asyncio
async def test_pre_call_extracts_multimodal_text_for_query():
    client = _client_recalling("m")
    logger = _logger_with(client)
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            }
        ],
        "metadata": {"session_id": "s1"},
    }

    await logger.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert client.recall.await_args.kwargs["query"] == "describe this"


@pytest.mark.asyncio
async def test_pre_call_swallows_client_errors_and_returns_data():
    client = MagicMock()
    client.recall = AsyncMock(side_effect=RuntimeError("dakera down"))
    logger = _logger_with(client)
    data = {"messages": [{"role": "user", "content": "hi"}], "metadata": {}}

    out = await logger.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert out is data  # never raises into the request path


@pytest.mark.asyncio
async def test_success_event_persists_user_and_assistant_exchange():
    client = _client_recalling()
    logger = _logger_with(client)
    response_obj = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="the answer"))])

    await logger.async_log_success_event(
        kwargs={
            "messages": [{"role": "user", "content": "the question"}],
            "metadata": {"session_id": "user-bob"},
            "model": "gpt-4o",
        },
        response_obj=response_obj,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    client.store_memory.assert_awaited_once()
    call = client.store_memory.await_args.kwargs
    assert call["agent_id"] == "user-bob"
    assert "User: the question" in call["content"]
    assert "Assistant: the answer" in call["content"]
    assert call["metadata"]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_success_event_skips_store_when_no_user_message():
    client = _client_recalling()
    logger = _logger_with(client)

    await logger.async_log_success_event(
        kwargs={"messages": [{"role": "system", "content": "sys only"}]},
        response_obj=SimpleNamespace(choices=[]),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    client.store_memory.assert_not_awaited()


def test_session_id_prefers_metadata_key():
    logger = DakeraMemoryLogger()
    assert logger._session_id({"session_id": "explicit"}) == "explicit"


def test_session_id_falls_back_to_stable_api_key_namespace():
    logger = DakeraMemoryLogger()
    auth = SimpleNamespace(api_key="sk-secret")

    first = logger._session_id(None, auth)
    second = logger._session_id({}, auth)

    assert first == second  # deterministic per key
    assert first.startswith("key:")
    assert "sk-secret" not in first  # raw key never leaked into the namespace


def test_session_id_defaults_when_no_identity():
    logger = DakeraMemoryLogger()
    assert logger._session_id(None, None) == "default"


def test_get_client_raises_helpful_error_when_sdk_missing(monkeypatch):
    # Simulate the optional 'dakera' package not being installed by masking it
    # in sys.modules; importing it then raises ImportError, which _get_client
    # must translate into an actionable install hint.
    monkeypatch.setitem(sys.modules, "dakera", None)
    logger = DakeraMemoryLogger()

    with pytest.raises(ImportError, match="pip install dakera"):
        logger._get_client()


def test_extract_text_handles_str_list_and_empty():
    assert _extract_text("plain") == "plain"
    assert _extract_text([{"type": "text", "text": "a"}, {"type": "image_url"}]) == "a"
    assert _extract_text(None) == ""


def test_base_url_defaults_to_local_dakera_port_3000(monkeypatch):
    monkeypatch.delenv("DAKERA_API_URL", raising=False)
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    logger = DakeraMemoryLogger(base_url=None, api_key=None)
    assert logger.base_url == "http://localhost:3000"
    assert logger.api_key is None
