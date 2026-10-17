"""Regression tests for #47005: /tag/new wrote the tag row before tagging its
models, so a failure mid-create 500'd and left an orphan row that blocked every
retry with "already exists". The create now rolls back: untag wherever the new
tag landed, then delete the row."""

import contextlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm.proxy.management_endpoints.tag_management_endpoints as endpoints
import litellm.proxy.proxy_server as proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.tag_management import TagNewRequest

TAG_NAME = "rollback-tag"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _deployment(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(model_info=SimpleNamespace(id=model_id))


def _make_table() -> SimpleNamespace:
    async def _create(data: Dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            tag_name=data["tag_name"],
            description=data["description"],
            models=data["models"],
            model_info=data["model_info"],
            created_at=NOW,
            updated_at=NOW,
            created_by=data["created_by"],
        )

    return SimpleNamespace(
        find_unique=AsyncMock(return_value=None),
        create=AsyncMock(side_effect=_create),
        delete=AsyncMock(),
    )


@contextlib.contextmanager
def _patched(table: SimpleNamespace, deployments: List[SimpleNamespace], add_side_effect):
    add_tag: AsyncMock = AsyncMock(side_effect=add_side_effect)
    remove_tag: AsyncMock = AsyncMock()
    with (
        patch.object(endpoints, "_table", return_value=table),
        patch.object(endpoints, "handle_budget_for_entity", AsyncMock(return_value=None)),
        patch.object(endpoints, "_get_model_names", AsyncMock(return_value={})),
        patch.object(endpoints, "_evict_tag_cache_keys", AsyncMock()),
        patch.object(endpoints, "get_deployments_by_model", AsyncMock(return_value=deployments)),
        patch.object(endpoints, "_add_tag_to_deployment", add_tag),
        patch.object(endpoints, "_remove_tag_from_deployment", remove_tag),
        patch.object(proxy_server, "prisma_client", MagicMock()),
        patch.object(proxy_server, "llm_router", MagicMock()),
        patch.object(proxy_server, "litellm_proxy_admin_name", "admin"),
    ):
        yield add_tag, remove_tag


@pytest.mark.asyncio
async def test_new_tag_rolls_back_row_and_deployments_when_tagging_fails():
    """One deployment write fails -> 500 to the caller, the tag is removed from
    every deployment it reached, and the fresh row is deleted so a retry works."""
    table = _make_table()
    deployments = [_deployment("model-1"), _deployment("model-2")]
    with _patched(
        table, deployments, [None, HTTPException(status_code=500, detail="db write failed")]
    ) as (add_tag, remove_tag):
        with pytest.raises(HTTPException) as exc_info:
            await endpoints.new_tag(
                tag=TagNewRequest(name=TAG_NAME, models=["model-a"]),
                user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
            )

    assert exc_info.value.status_code == 500
    # every deployment write was attempted (return_exceptions gather)
    assert add_tag.await_count == 2
    # rollback: untag everywhere the new tag could have landed...
    assert remove_tag.await_count == 2
    assert sorted(id(c.kwargs["deployment"]) for c in remove_tag.await_args_list) == sorted(id(d) for d in deployments)
    assert all(c.kwargs["tag"] == TAG_NAME for c in remove_tag.await_args_list)
    # ...then delete the orphaned row
    table.delete.assert_awaited_once_with(where={"tag_name": TAG_NAME})


@pytest.mark.asyncio
async def test_new_tag_success_path_does_not_rollback():
    table = _make_table()
    deployments = [_deployment("model-1")]
    with _patched(table, deployments, [None]) as (add_tag, remove_tag):
        resp = await endpoints.new_tag(
            tag=TagNewRequest(name=TAG_NAME, models=["model-a"]),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )

    assert resp["message"] == f"Tag {TAG_NAME} created successfully"
    add_tag.assert_awaited_once()
    remove_tag.assert_not_awaited()
    table.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_remove_tag_from_deployment_removes_only_that_tag():
    table = SimpleNamespace(
        find_unique=AsyncMock(
            return_value=SimpleNamespace(
                litellm_params=json.dumps({"tags": [TAG_NAME, "other-tag"], "api_key": "encrypted-value"})
            )
        ),
        update=AsyncMock(),
    )
    with (
        patch.object(endpoints, "ModelRepository", return_value=SimpleNamespace(table=table)),
        patch.object(proxy_server, "prisma_client", MagicMock()),
    ):
        await endpoints._remove_tag_from_deployment(deployment=_deployment("model-1"), tag=TAG_NAME)

    table.update.assert_awaited_once()
    written = json.loads(table.update.await_args.kwargs["data"]["litellm_params"])
    assert written["tags"] == ["other-tag"]
    assert written["api_key"] == "encrypted-value"


@pytest.mark.asyncio
async def test_remove_tag_from_deployment_noops_when_tag_absent():  # test-quality-ok: the no-op contract is the absence of a DB write, so assert_not_awaited on table.update is the caller-observable assertion
    table = SimpleNamespace(
        find_unique=AsyncMock(return_value=SimpleNamespace(litellm_params=json.dumps({"tags": ["other-tag"]}))),
        update=AsyncMock(),
    )
    with (
        patch.object(endpoints, "ModelRepository", return_value=SimpleNamespace(table=table)),
        patch.object(proxy_server, "prisma_client", MagicMock()),
    ):
        await endpoints._remove_tag_from_deployment(deployment=_deployment("model-1"), tag=TAG_NAME)

    table.update.assert_not_awaited()
