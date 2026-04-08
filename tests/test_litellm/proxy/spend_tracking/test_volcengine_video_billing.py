"""
Tests for Volcengine video billing module.
"""

import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import litellm

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

# Mock prisma.Json before importing the module under test
import prisma


class MockPrismaJson:
    """Mock for prisma.Json that mimics the real class behavior."""

    def __init__(self, data):
        self.data = data


prisma.Json = MockPrismaJson  # type: ignore[misc, assignment]

from litellm.proxy.spend_tracking.volcengine_video_billing import (
    VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD,
    VolcengineVideoBillingManager,
)
from litellm._logging import verbose_proxy_logger
from litellm.types.videos.main import VideoObject


def _build_manager() -> VolcengineVideoBillingManager:
    prisma_client = MagicMock()
    prisma_client.db.litellm_videotasktable.upsert = AsyncMock()
    prisma_client.db.litellm_videotasktable.update_many = AsyncMock()
    prisma_client.db.litellm_videotasktable.update = AsyncMock()
    prisma_client.db.litellm_videotasktable.find_unique = AsyncMock()
    prisma_client.db.litellm_spendlogs.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_spendlogs.upsert = AsyncMock()
    prisma_client.db.litellm_spendlogs.update = AsyncMock()
    return VolcengineVideoBillingManager(
        prisma_client=prisma_client,
        llm_router=MagicMock(),
        db_spend_update_writer=MagicMock(),
        proxy_logging_obj=MagicMock(),
    )


def _build_generation_kwargs(base_model: str) -> dict:
    return {
        "call_type": "avideo_generation",
        "custom_llm_provider": "volcengine",
        "model": "seedance-2-video",
        "litellm_params": {
            "metadata": {
                "user_api_key": "sk-volc-test",
                "user_api_key_user_id": "user-1",
                "user_api_key_team_id": "team-1",
                "user_api_key_org_id": "org-1",
                "user_api_key_end_user_id": "end-user-1",
                "model_info": {
                    "id": "deployment-123",
                    "base_model": base_model,
                },
                "tags": ["video-billing"],
            }
        },
        "standard_logging_object": {
            "metadata": {"user_api_key_hash": "hashed-key-123"},
            "request_tags": ["standard-tag"],
            "response_cost": 99.0,
        },
    }


@pytest.fixture
def patched_volcengine_model_cost():
    with patch.dict(
        litellm.model_cost,
        {
            "volcengine/doubao-seedance-2.0": {
                "provider_pricing_currency": "CNY",
                "volcengine_video_output_cost_per_million_tokens_without_input_video": 46.0,
                "volcengine_video_output_cost_per_million_tokens_with_input_video": 28.0,
            }
        },
        clear=False,
    ):
        yield


@pytest.mark.asyncio
async def test_register_pending_video_task_uses_versionless_pricing_without_input_video(
    patched_volcengine_model_cost,
):
    manager = _build_manager()
    kwargs = _build_generation_kwargs(base_model="doubao-seedance-2.0-260128")
    response = VideoObject(
        id="video_test_123",
        object="video",
        status="queued",
        model="ep-20260402174450-9qflb",
        seconds="11",
        usage={"duration_seconds": 11.0},
    )
    response._hidden_params = {
        "request_content": [
            {"type": "text", "text": "Create a cinematic fruit tea ad"},
        ]
    }

    overridden_cost = await manager.handle_success_event(
        kwargs=kwargs,
        completion_response=response,
    )

    assert overridden_cost == 0.0
    assert kwargs["response_cost"] == 0.0
    assert kwargs["standard_logging_object"]["response_cost"] == 0.0

    manager.prisma_client.db.litellm_videotasktable.upsert.assert_awaited_once()
    upsert_data = (
        manager.prisma_client.db.litellm_videotasktable.upsert.call_args.kwargs["data"][
            "create"
        ]
    )
    assert upsert_data["pricing_model"] == "volcengine/doubao-seedance-2.0-260128"
    assert upsert_data["price_per_million_tokens"] == 46.0
    assert upsert_data["pricing_currency"] == "CNY"
    assert upsert_data["has_input_video"] is False
    assert upsert_data["api_key"] == "hashed-key-123"
    assert isinstance(upsert_data["request_tags"], prisma.Json)
    assert upsert_data["request_tags"].data == ["standard-tag"]
    assert isinstance(upsert_data["metadata"], prisma.Json)
    assert upsert_data["metadata"].data == {
        "request_content": [
            {"type": "text", "text": "Create a cinematic fruit tea ad"},
        ]
    }


@pytest.mark.asyncio
async def test_register_pending_video_task_uses_input_video_price(
    patched_volcengine_model_cost,
):
    manager = _build_manager()
    kwargs = _build_generation_kwargs(base_model="doubao-seedance-2.0-260128")
    response = VideoObject(
        id="video_test_456",
        object="video",
        status="queued",
        model="ep-20260402174450-9qflb",
        seconds="11",
        usage={"duration_seconds": 11.0},
    )
    response._hidden_params = {
        "request_content": [
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/reference.mp4"},
                "role": "reference_video",
            }
        ]
    }

    await manager.handle_success_event(kwargs=kwargs, completion_response=response)

    upsert_data = (
        manager.prisma_client.db.litellm_videotasktable.upsert.call_args.kwargs["data"][
            "create"
        ]
    )
    assert upsert_data["price_per_million_tokens"] == 28.0
    assert upsert_data["has_input_video"] is True


@pytest.mark.asyncio
async def test_register_pending_video_task_auto_registers_runtime_pricing_models():
    manager = _build_manager()
    kwargs = _build_generation_kwargs(base_model="doubao-seedance-2.0-260128")
    response = VideoObject(
        id="video_test_auto_register",
        object="video",
        status="queued",
        model="ep-20260402174450-9qflb",
        seconds="11",
        usage={"duration_seconds": 11.0},
    )
    response._hidden_params = {
        "request_content": [
            {"type": "text", "text": "Create a cinematic fruit tea ad"},
        ]
    }

    with patch.dict(litellm.model_cost, {}, clear=True):
        await manager.handle_success_event(kwargs=kwargs, completion_response=response)

        assert "volcengine/doubao-seedance-2.0" in litellm.model_cost
        upsert_data = (
            manager.prisma_client.db.litellm_videotasktable.upsert.call_args.kwargs[
                "data"
            ]["create"]
        )
        assert upsert_data["price_per_million_tokens"] == 46.0
        assert upsert_data["pricing_currency"] == "CNY"


@pytest.mark.asyncio
async def test_register_pending_video_task_accepts_dict_response_and_create_video_alias(
    patched_volcengine_model_cost,
):
    manager = _build_manager()
    kwargs = _build_generation_kwargs(base_model="doubao-seedance-2.0-260128")
    kwargs["call_type"] = "create_video"
    response = {
        "id": "video_test_dict_create",
        "object": "video",
        "status": "queued",
        "model": "ep-20260402174450-9qflb",
        "seconds": "5",
        "usage": {"duration_seconds": 5.0},
    }

    overridden_cost = await manager.handle_success_event(
        kwargs=kwargs,
        completion_response=response,
    )

    assert overridden_cost == 0.0
    manager.prisma_client.db.litellm_videotasktable.upsert.assert_awaited_once()
    upsert_data = (
        manager.prisma_client.db.litellm_videotasktable.upsert.call_args.kwargs["data"][
            "create"
        ]
    )
    assert upsert_data["video_id"] == "video_test_dict_create"
    assert upsert_data["provider_model"] == "ep-20260402174450-9qflb"
    assert upsert_data["price_per_million_tokens"] == 46.0


@pytest.mark.asyncio
async def test_register_pending_video_task_rebuilds_request_content_from_existing_spend_log(
    patched_volcengine_model_cost,
):
    manager = _build_manager()
    kwargs = _build_generation_kwargs(base_model="doubao-seedance-2.0-260128")
    response = VideoObject(
        id="video_test_rebuild_request_content",
        object="video",
        status="queued",
        model="ep-20260402174450-9qflb",
        seconds="5",
        usage={"duration_seconds": 5.0},
    )
    manager.prisma_client.db.litellm_spendlogs.find_unique.return_value = (
        SimpleNamespace(
            proxy_server_request={
                "prompt": "Create a cinematic fruit tea ad",
                "seconds": "5",
                "size": "1280x720",
            }
        )
    )

    await manager._register_pending_video_task(
        kwargs=kwargs,
        completion_response=response,
    )

    upsert_data = (
        manager.prisma_client.db.litellm_videotasktable.upsert.call_args.kwargs["data"][
            "create"
        ]
    )
    assert isinstance(upsert_data["metadata"], prisma.Json)
    assert upsert_data["metadata"].data["request_content"] == [
        {"type": "text", "text": "Create a cinematic fruit tea ad"}
    ]
    assert upsert_data["has_input_video"] is False


@pytest.mark.asyncio
async def test_register_pending_video_task_prefers_existing_spend_log_identity(
    patched_volcengine_model_cost,
):
    manager = _build_manager()
    kwargs = _build_generation_kwargs(base_model="doubao-seedance-2.0-260128")
    kwargs["standard_logging_object"]["metadata"]["user_api_key_hash"] = "wrong-hash"
    response = VideoObject(
        id="video_test_existing_identity",
        object="video",
        status="queued",
        model="ep-20260402174450-9qflb",
        seconds="5",
        usage={"duration_seconds": 5.0},
    )
    response._hidden_params = {
        "request_content": [
            {"type": "text", "text": "Create a cinematic fruit tea ad"},
        ]
    }
    manager.prisma_client.db.litellm_spendlogs.find_unique.return_value = (
        SimpleNamespace(
            api_key="original-hash-123",
            user="original-user",
            team_id="original-team",
            organization_id="original-org",
            end_user="original-end-user",
        )
    )

    await manager._register_pending_video_task(
        kwargs=kwargs,
        completion_response=response,
    )

    upsert_data = (
        manager.prisma_client.db.litellm_videotasktable.upsert.call_args.kwargs["data"][
            "create"
        ]
    )
    assert upsert_data["api_key"] == "original-hash-123"
    assert upsert_data["user"] == "original-user"
    assert upsert_data["team_id"] == "original-team"
    assert upsert_data["organization_id"] == "original-org"
    assert upsert_data["end_user"] == "original-end-user"


@pytest.mark.asyncio
async def test_reconcile_task_from_video_response_reregisters_missing_task_when_kwargs_provided():
    manager = _build_manager()
    manager._register_pending_video_task = AsyncMock()
    manager._finalize_completed_task = AsyncMock()
    task = SimpleNamespace(video_id="video_test_reregister", billing_state="pending")
    manager.prisma_client.db.litellm_videotasktable.find_unique = AsyncMock(
        side_effect=[None, task]
    )
    kwargs = _build_generation_kwargs(base_model="doubao-seedance-2.0-260128")
    response = VideoObject(
        id="video_test_reregister",
        object="video",
        status="completed",
        model="ep-20260402174450-9qflb",
        usage={"total_tokens": 1000, "completion_tokens": 1000},
    )

    await manager._reconcile_task_from_video_response(
        video_id="video_test_reregister",
        video_response=response,
        kwargs=kwargs,
    )

    manager._register_pending_video_task.assert_awaited_once_with(
        kwargs=kwargs,
        completion_response=response,
    )
    manager._finalize_completed_task.assert_awaited_once_with(
        task=task,
        video_response=response,
    )


@pytest.mark.asyncio
async def test_finalize_completed_task_bills_from_provider_total_tokens():
    manager = _build_manager()
    manager._apply_async_billing_delta = AsyncMock()
    manager._upsert_final_spend_log = AsyncMock()
    manager.prisma_client.db.litellm_videotasktable.update_many.return_value = 1

    task = SimpleNamespace(
        video_id="video_test_789",
        billing_state="pending",
        price_per_million_tokens=46.0,
        pricing_currency="CNY",
        spend=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        created_at=datetime.now(timezone.utc),
        api_key="hashed-key-123",
        user="user-1",
        team_id="team-1",
        organization_id="org-1",
        end_user="end-user-1",
        model="seedance-2-video",
        model_group="seedance-2-video",
        model_id="deployment-123",
        request_tags=["video-billing"],
        custom_llm_provider="volcengine",
    )
    video_response = VideoObject(
        id=task.video_id,
        object="video",
        status="completed",
        completed_at=1775549339,
        model="ep-20260402174450-9qflb",
        seconds="11",
        size="16:9",
        usage={
            "total_tokens": 238500,
            "completion_tokens": 238500,
            "duration_seconds": 11.0,
        },
    )

    await manager._finalize_completed_task(task=task, video_response=video_response)

    expected_provider_spend = 238500 / 1_000_000 * 46.0
    expected_spend = expected_provider_spend / VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD
    manager._apply_async_billing_delta.assert_awaited_once()
    delta_call = manager._apply_async_billing_delta.call_args.kwargs
    assert delta_call["delta_spend"] == pytest.approx(expected_spend)
    assert delta_call["provider_delta_spend"] == pytest.approx(expected_provider_spend)
    assert delta_call["delta_prompt_tokens"] == 0
    assert delta_call["delta_completion_tokens"] == 238500

    manager._upsert_final_spend_log.assert_awaited_once()
    upsert_call = manager._upsert_final_spend_log.call_args.kwargs
    assert upsert_call["final_spend"] == pytest.approx(expected_spend)
    assert upsert_call["provider_spend_amount"] == pytest.approx(
        expected_provider_spend
    )
    assert upsert_call["total_tokens"] == 238500
    assert upsert_call["completion_tokens"] == 238500
    update_data = (
        manager.prisma_client.db.litellm_videotasktable.update.call_args.kwargs["data"]
    )
    assert update_data["billing_state"] == "billed"
    assert update_data["spend"] == pytest.approx(expected_spend)
    assert update_data["total_tokens"] == 238500
    assert update_data["completion_tokens"] == 238500
    assert update_data["duration_seconds"] == 11.0


@pytest.mark.asyncio
async def test_upsert_final_spend_log_wraps_json_fields_for_prisma():
    manager = _build_manager()
    task = SimpleNamespace(
        video_id="video_test_spendlog_json",
        api_key="hashed-key-123",
        user="user-1",
        team_id="team-1",
        organization_id="org-1",
        end_user="end-user-1",
        model="seedance-2-video",
        model_group="seedance-2-video",
        model_id="deployment-123",
        custom_llm_provider="volcengine",
        pricing_currency="CNY",
        request_tags=["video-billing"],
        created_at=datetime.now(timezone.utc),
    )
    manager.prisma_client.db.litellm_spendlogs.find_unique.return_value = (
        SimpleNamespace(
            request_id=task.video_id,
            metadata={"existing": True},
        )
    )

    await manager._upsert_final_spend_log(
        task=task,
        final_spend=0.6944444444,
        provider_spend_amount=5.0,
        total_tokens=1000,
        prompt_tokens=0,
        completion_tokens=1000,
        usage={"total_tokens": 1000},
    )

    manager.prisma_client.db.litellm_spendlogs.update.assert_awaited_once()
    update_data = manager.prisma_client.db.litellm_spendlogs.update.call_args.kwargs[
        "data"
    ]
    assert isinstance(update_data["metadata"], prisma.Json)
    assert update_data["metadata"].data["existing"] is True
    assert update_data["metadata"].data["provider_spend_amount"] == 5.0
    assert update_data["metadata"].data["billing_spend_currency"] == "USD"
    assert update_data["metadata"].data["billing_spend_amount"] == pytest.approx(
        0.6944444444
    )


def test_build_final_spend_log_metadata_preserves_existing_dict_metadata():
    manager = _build_manager()
    provider_spend_amount = 10.971
    billing_spend_amount = provider_spend_amount / VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD

    metadata = manager._build_final_spend_log_metadata(
        existing_metadata={"existing_key": "existing_value"},
        usage={"total_tokens": 238500},
        pricing_currency="CNY",
        provider_spend_amount=provider_spend_amount,
        final_spend=billing_spend_amount,
        video_task_id="video_test_789",
    )

    assert metadata["existing_key"] == "existing_value"
    assert metadata["usage_object"] == {"total_tokens": 238500}
    assert metadata["provider_spend_currency"] == "CNY"
    assert metadata["provider_spend_amount"] == pytest.approx(provider_spend_amount)
    assert metadata["billing_spend_currency"] == "USD"
    assert metadata["billing_spend_amount"] == pytest.approx(billing_spend_amount)
    assert metadata["video_billing_task_id"] == "video_test_789"


@pytest.mark.skip(
    reason="Requires full proxy_server dependencies - tested via integration tests"
)
@pytest.mark.asyncio
async def test_apply_async_billing_delta_prefers_existing_spend_log_identity():
    """This test requires proxy_server dependencies (orjson, fastapi, etc).
    Tested via integration tests instead.
    """
    pass


@pytest.mark.asyncio
async def test_poll_pending_video_tasks_disables_itself_when_prisma_client_missing_table():
    prisma_client = MagicMock()
    prisma_client.db = SimpleNamespace()
    manager = VolcengineVideoBillingManager(
        prisma_client=prisma_client,
        llm_router=MagicMock(),
        db_spend_update_writer=MagicMock(),
        proxy_logging_obj=MagicMock(),
    )

    with patch.object(verbose_proxy_logger, "warning") as warning_mock:
        await manager.poll_pending_video_tasks()
        await manager.poll_pending_video_tasks()

    assert manager._video_task_table_unavailable is True
    warning_mock.assert_called_once()
