from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.model_usage_rollup import increment_daily_model_usage, model_usage_task_type


def test_model_usage_task_type_maps_supported_calls() -> None:
    assert model_usage_task_type("acompletion") == "chat"
    assert model_usage_task_type("aembedding") == "embeddings"
    assert model_usage_task_type("not-a-real-call") == "unknown"


@pytest.mark.asyncio
async def test_increment_daily_model_usage_uses_atomic_prisma_upsert() -> None:
    table = MagicMock()
    table.upsert = AsyncMock()
    prisma_client = MagicMock()
    prisma_client.db.litellm_dailymodelusage = table
    payload = {
        "request_id": "request-1",
        "call_type": "acompletion",
        "api_key": "key",
        "spend": 0.25,
        "total_tokens": 30,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "startTime": datetime(2026, 9, 28, tzinfo=timezone.utc),
        "endTime": datetime(2026, 9, 28, tzinfo=timezone.utc),
        "completionStartTime": None,
        "model": "openai/gpt-5.4-mini",
        "model_id": None,
        "model_group": "fast-chat",
        "mcp_namespaced_tool_name": None,
        "agent_id": None,
        "api_base": "",
        "user": "user",
        "metadata": "{}",
        "cache_hit": "False",
        "cache_key": "",
        "request_tags": "[]",
        "team_id": None,
        "organization_id": None,
        "end_user": None,
        "requester_ip_address": None,
        "custom_llm_provider": "openai",
        "messages": None,
        "response": None,
        "proxy_server_request": None,
        "session_id": None,
        "request_duration_ms": 20,
        "status": "success",
        "litellm_call_id": None,
    }

    await increment_daily_model_usage(prisma_client, payload)

    call = table.upsert.await_args.kwargs
    assert call["data"]["create"]["request_count"] == 1
    assert call["data"]["update"]["completion_tokens"] == {"increment": 20}
    assert call["data"]["create"]["task_type"] == "chat"
