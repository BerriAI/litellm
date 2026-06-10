import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.integrations.faros.faros_logger import USAGE_MUTATION, FarosLogger
from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry


def make_logger(mock_post=None, **kwargs) -> FarosLogger:
    client = MagicMock()
    client.post = mock_post or AsyncMock()
    return FarosLogger(api_key="test-key", async_httpx_client=client, **kwargs)


def make_response(body=None):
    response = MagicMock()
    response.json.return_value = body if body is not None else {"data": {}}
    return response


def make_kwargs(
    startTime: float = 1765000000.5,
    user_email=None,
    user_id=None,
    end_user=None,
) -> dict:
    return {
        "standard_logging_object": {
            "startTime": startTime,
            "end_user": end_user,
            "metadata": {
                "user_api_key_user_email": user_email,
                "user_api_key_user_id": user_id,
            },
        }
    }


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("FAROS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FAROS_API_KEY"):
        FarosLogger()


def test_init_without_running_event_loop():
    logger = make_logger()
    assert logger.flush_lock is not None


@pytest.mark.asyncio
async def test_user_uid_resolution_priority():
    logger = make_logger()

    await logger.async_log_success_event(
        make_kwargs(user_email="dev@example.com", user_id="user-1", end_user="cust"),
        None,
        None,
        None,
    )
    await logger.async_log_success_event(
        make_kwargs(user_id="user-1", end_user="cust"), None, None, None
    )
    await logger.async_log_success_event(make_kwargs(end_user="cust"), None, None, None)
    await logger.async_log_success_event(make_kwargs(), None, None, None)

    assert [record["user_uid"] for record in logger.log_queue] == [
        "dev@example.com",
        "user-1",
        "cust",
        "unknown",
    ]


@pytest.mark.asyncio
async def test_proxy_admin_placeholder_is_not_a_user():
    from litellm.constants import LITELLM_PROXY_ADMIN_NAME

    logger = make_logger()

    await logger.async_log_success_event(
        make_kwargs(user_id=LITELLM_PROXY_ADMIN_NAME, end_user="cust"),
        None,
        None,
        None,
    )
    await logger.async_log_success_event(
        make_kwargs(user_id=LITELLM_PROXY_ADMIN_NAME), None, None, None
    )

    assert [record["user_uid"] for record in logger.log_queue] == ["cust", "unknown"]


@pytest.mark.asyncio
async def test_proxy_admin_placeholder_user_id_is_ignored():
    logger = make_logger()
    await logger.async_log_success_event(
        make_kwargs(user_id="default_user_id", end_user="dev@example.com"),
        None,
        None,
        None,
    )
    assert logger.log_queue[0]["user_uid"] == "dev@example.com"


@pytest.mark.asyncio
async def test_send_batch_posts_usage_mutation():
    mock_post = AsyncMock(return_value=make_response())
    logger = make_logger(mock_post=mock_post, graph="my-graph")

    await logger.async_log_success_event(
        make_kwargs(startTime=1765000000.5, user_email="dev@example.com"),
        None,
        None,
        None,
    )
    await logger.async_send_batch()

    mock_post.assert_awaited_once()
    call = mock_post.call_args
    assert call.kwargs["url"] == "https://prod.api.faros.ai/graphs/my-graph/graphql"
    assert call.kwargs["headers"]["authorization"] == "test-key"

    body = call.kwargs["json"]
    assert body["query"] == USAGE_MUTATION
    assert "insert_vcs_UserToolUsage" in body["query"]
    usages = body["variables"]["usages"]
    assert len(usages) == 1
    usage = usages[0]
    assert usage["usedAt"] == "2025-12-06T05:46:40.500+00:00"
    assert usage["origin"] == "litellm"
    user_tool = usage["userTool"]["data"]
    assert user_tool["tool"] == {"category": "LiteLLM"}
    assert user_tool["user"]["data"] == {
        "uid": "dev@example.com",
        "source": "LiteLLM",
        "origin": "litellm",
    }
    assert user_tool["user"]["on_conflict"]["constraint"] == "vcs_User_pkey"
    assert usage["userTool"]["on_conflict"]["constraint"] == "vcs_UserTool_pkey"
    json.dumps(body)


@pytest.mark.asyncio
async def test_send_batch_dedupes_rows_with_same_primary_key():
    mock_post = AsyncMock(return_value=make_response())
    logger = make_logger(mock_post=mock_post)

    for _ in range(2):
        await logger.async_log_success_event(
            make_kwargs(startTime=1765000000.5, user_id="user-1"), None, None, None
        )
    await logger.async_log_success_event(
        make_kwargs(startTime=1765000000.5, user_id="user-2"), None, None, None
    )
    await logger.async_send_batch()

    usages = mock_post.call_args.kwargs["json"]["variables"]["usages"]
    assert len(usages) == 2
    assert {u["userTool"]["data"]["user"]["data"]["uid"] for u in usages} == {
        "user-1",
        "user-2",
    }


@pytest.mark.asyncio
async def test_graphql_errors_raise_and_preserve_queue():
    mock_post = AsyncMock(
        return_value=make_response({"errors": [{"message": "unknown field"}]})
    )
    logger = make_logger(mock_post=mock_post)

    await logger.async_log_success_event(
        make_kwargs(user_id="user-1"), None, None, None
    )
    with pytest.raises(ValueError, match="unknown field"):
        await logger.async_send_batch()

    await logger.flush_queue()
    assert len(logger.log_queue) == 1


@pytest.mark.asyncio
async def test_batch_size_triggers_flush():
    mock_post = AsyncMock(return_value=make_response())
    logger = make_logger(mock_post=mock_post, batch_size=2)

    await logger.async_log_success_event(
        make_kwargs(startTime=1765000000.5, user_id="user-1"), None, None, None
    )
    mock_post.assert_not_awaited()
    await logger.async_log_success_event(
        make_kwargs(startTime=1765000001.5, user_id="user-1"), None, None, None
    )
    mock_post.assert_awaited_once()
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_event_without_standard_logging_object_is_skipped():
    logger = make_logger()
    await logger.async_log_success_event({}, None, None, None)
    assert logger.log_queue == []


def test_faros_is_a_registered_callback():
    assert CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE["faros"] is FarosLogger
    assert "faros" in litellm._known_custom_logger_compatible_callbacks


@pytest.mark.asyncio
async def test_init_custom_logger_compatible_class_returns_singleton(monkeypatch):
    from litellm.litellm_core_utils.litellm_logging import (
        _in_memory_loggers,
        _init_custom_logger_compatible_class,
        get_custom_logger_compatible_class,
    )

    monkeypatch.setenv("FAROS_API_KEY", "test-key")
    assert get_custom_logger_compatible_class("faros") is None
    created = _init_custom_logger_compatible_class("faros", None, None)
    try:
        assert isinstance(created, FarosLogger)
        again = _init_custom_logger_compatible_class("faros", None, None)
        assert again is created
        assert get_custom_logger_compatible_class("faros") is created
    finally:
        _in_memory_loggers.remove(created)


@pytest.mark.asyncio
async def test_send_batch_with_empty_queue_does_not_post():
    mock_post = AsyncMock()
    logger = make_logger(mock_post=mock_post)
    await logger.async_send_batch()
    mock_post.assert_not_awaited()
