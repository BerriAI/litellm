"""
DB-helper coverage for nested access groups (#28032).

Uses a hand-rolled fake Prisma client to exercise the async helpers without
spinning up Postgres. Pairs with the pure-function tests in
test_nested_access_groups.py.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
    _clear_access_group_from_all_deployments,
    _dual_write_group_membership,
    delete_group_membership_edges,
    get_all_access_groups_from_db,
    get_group_memberships_from_db,
    upsert_group_memberships,
)


def _make_prisma(membership_rows=None, deployment_rows=None):
    """Build a fake PrismaClient exposing only the tables this PR touches."""
    membership_rows = membership_rows or []
    deployment_rows = deployment_rows or []

    db = MagicMock()
    db.litellm_accessgroupmembership = MagicMock()
    db.litellm_accessgroupmembership.find_many = AsyncMock(return_value=membership_rows)
    db.litellm_accessgroupmembership.create_many = AsyncMock(return_value=0)
    db.litellm_accessgroupmembership.delete_many = AsyncMock(return_value=0)
    db.litellm_proxymodeltable = MagicMock()
    db.litellm_proxymodeltable.find_many = AsyncMock(return_value=deployment_rows)
    db.litellm_proxymodeltable.update = AsyncMock(return_value=None)

    client = MagicMock()
    client.db = db
    return client


def _row(parent: str, child: str) -> SimpleNamespace:
    return SimpleNamespace(parent_group=parent, child_group=child)


def _deployment(model_id: str, model_name: str, access_groups):
    return SimpleNamespace(
        model_id=model_id,
        model_name=model_name,
        model_info={"access_groups": list(access_groups)},
    )


# ---------------------------------------------------------------------------
# get_group_memberships_from_db
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_group_memberships_buckets_by_parent():
    """Membership rows are bucketed parent -> [children] in a single query."""
    prisma = _make_prisma(
        membership_rows=[
            _row("project-x", "image"),
            _row("project-x", "reasoning"),
            _row("project-y", "image"),
        ]
    )
    result = await get_group_memberships_from_db(prisma_client=prisma)
    assert result == {
        "project-x": ["image", "reasoning"],
        "project-y": ["image"],
    }
    prisma.db.litellm_accessgroupmembership.find_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_group_memberships_empty_table_returns_empty_dict():
    prisma = _make_prisma(membership_rows=[])
    assert await get_group_memberships_from_db(prisma_client=prisma) == {}


# ---------------------------------------------------------------------------
# upsert_group_memberships
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_group_memberships_empty_child_list_is_noop():
    """Empty child list short-circuits before touching the DB."""
    prisma = _make_prisma()
    n = await upsert_group_memberships(
        parent_group="A", child_groups=[], prisma_client=prisma
    )
    assert n == 0
    prisma.db.litellm_accessgroupmembership.create_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_group_memberships_self_reference_rejected():
    """parent == child must 400 without touching the DB."""
    prisma = _make_prisma()
    with pytest.raises(HTTPException) as exc:
        await upsert_group_memberships(
            parent_group="A", child_groups=["A", "B"], prisma_client=prisma
        )
    assert exc.value.status_code == 400
    assert "cannot include itself" in exc.value.detail["error"]
    prisma.db.litellm_accessgroupmembership.create_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_group_memberships_uses_create_many_with_skip_duplicates():
    """Normal path issues one batch create_many with skip_duplicates=True."""
    prisma = _make_prisma()
    prisma.db.litellm_accessgroupmembership.create_many = AsyncMock(return_value=2)
    n = await upsert_group_memberships(
        parent_group="project-x",
        child_groups=["image", "reasoning"],
        prisma_client=prisma,
    )
    assert n == 2
    call = prisma.db.litellm_accessgroupmembership.create_many.await_args
    assert call.kwargs["skip_duplicates"] is True
    assert call.kwargs["data"] == [
        {"parent_group": "project-x", "child_group": "image"},
        {"parent_group": "project-x", "child_group": "reasoning"},
    ]


# ---------------------------------------------------------------------------
# delete_group_membership_edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_group_membership_edges_targets_both_sides():
    """Delete must match rows where the group appears as parent OR child."""
    prisma = _make_prisma()
    prisma.db.litellm_accessgroupmembership.delete_many = AsyncMock(return_value=4)
    n = await delete_group_membership_edges(
        access_group="project-x", prisma_client=prisma
    )
    assert n == 4
    call = prisma.db.litellm_accessgroupmembership.delete_many.await_args
    assert call.kwargs["where"] == {
        "OR": [
            {"parent_group": "project-x"},
            {"child_group": "project-x"},
        ]
    }


# ---------------------------------------------------------------------------
# _clear_access_group_from_all_deployments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_access_group_only_touches_deployments_carrying_it():
    """Deployments without the tag must not generate UPDATE calls."""
    prisma = _make_prisma(
        deployment_rows=[
            _deployment("m1", "gpt-4", ["production", "image"]),
            _deployment("m2", "claude-3", ["other"]),
            _deployment("m3", "dall-e-3", ["image"]),
        ]
    )
    touched = await _clear_access_group_from_all_deployments(
        access_group="image", prisma_client=prisma
    )
    assert touched == 2
    # Two updates: m1 and m3 (m2 had no "image" tag)
    assert prisma.db.litellm_proxymodeltable.update.await_count == 2


@pytest.mark.asyncio
async def test_clear_access_group_no_match_is_noop():
    """No deployment carries the tag -> no UPDATE calls, return 0."""
    prisma = _make_prisma(deployment_rows=[_deployment("m1", "gpt-4", ["other"])])
    touched = await _clear_access_group_from_all_deployments(
        access_group="ghost", prisma_client=prisma
    )
    assert touched == 0
    prisma.db.litellm_proxymodeltable.update.assert_not_awaited()


# ---------------------------------------------------------------------------
# _dual_write_group_membership
# ---------------------------------------------------------------------------


class _FakeRouter:
    def __init__(self, model_names):
        self._names = model_names

    def get_model_names(self):
        return self._names


@pytest.mark.asyncio
async def test_dual_write_routes_real_models_to_deployment_path():
    """Real model names go through update_deployments_with_access_group only."""
    prisma = _make_prisma(deployment_rows=[_deployment("m1", "gpt-4", [])])
    writes = await _dual_write_group_membership(
        access_group="production",
        member_names=["gpt-4"],
        known_access_groups=set(),
        llm_router=_FakeRouter(["gpt-4"]),
        prisma_client=prisma,
    )
    assert writes == 1
    prisma.db.litellm_accessgroupmembership.create_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_dual_write_routes_child_groups_to_membership_table():
    """Group names go through upsert_group_memberships only."""
    prisma = _make_prisma()
    prisma.db.litellm_accessgroupmembership.create_many = AsyncMock(return_value=1)
    writes = await _dual_write_group_membership(
        access_group="project-x",
        member_names=["image"],
        known_access_groups={"image"},
        llm_router=_FakeRouter(["gpt-4"]),
        prisma_client=prisma,
    )
    assert writes == 1
    prisma.db.litellm_proxymodeltable.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_dual_write_mixed_input_hits_both_paths():
    """Real models + child groups in the same call sum write counts from both paths."""
    prisma = _make_prisma(deployment_rows=[_deployment("m1", "gpt-4", [])])
    prisma.db.litellm_accessgroupmembership.create_many = AsyncMock(return_value=1)
    writes = await _dual_write_group_membership(
        access_group="project-x",
        member_names=["gpt-4", "image"],
        known_access_groups={"image"},
        llm_router=_FakeRouter(["gpt-4"]),
        prisma_client=prisma,
    )
    assert writes == 2  # 1 deployment touched + 1 edge added
    prisma.db.litellm_proxymodeltable.update.assert_awaited()
    prisma.db.litellm_accessgroupmembership.create_many.assert_awaited()


@pytest.mark.asyncio
async def test_dual_write_with_null_router_treats_everything_as_unknown_or_group():
    """No router -> known_access_groups is the only positive classification."""
    prisma = _make_prisma()
    prisma.db.litellm_accessgroupmembership.create_many = AsyncMock(return_value=1)
    writes = await _dual_write_group_membership(
        access_group="project-x",
        member_names=["image"],
        known_access_groups={"image"},
        llm_router=None,
        prisma_client=prisma,
    )
    assert writes == 1


# ---------------------------------------------------------------------------
# get_all_access_groups_from_db
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_access_groups_returns_direct_tags_when_no_memberships():
    """Backwards-compat: with no membership rows, behavior matches the flat model."""
    prisma = _make_prisma(
        deployment_rows=[
            _deployment("m1", "gpt-4", ["production"]),
            _deployment("m2", "claude-3", ["production"]),
            _deployment("m3", "dall-e-3", ["image"]),
        ]
    )
    result = await get_all_access_groups_from_db(prisma_client=prisma)
    assert set(result.keys()) == {"production", "image"}
    assert result["production"].model_names == ["claude-3", "gpt-4"]
    assert result["production"].deployment_count == 2
    assert result["production"].parent_groups == []
    assert result["production"].child_groups == []
    assert result["image"].model_names == ["dall-e-3"]


@pytest.mark.asyncio
async def test_get_all_access_groups_expands_nested_model_names():
    """Nested groups: parent's model_names includes the child's models."""
    prisma = _make_prisma(
        deployment_rows=[
            _deployment("m1", "gpt-4", ["reasoning"]),
            _deployment("m2", "dall-e-3", ["image"]),
        ],
        membership_rows=[
            _row("project-x", "image"),
            _row("project-x", "reasoning"),
        ],
    )
    result = await get_all_access_groups_from_db(prisma_client=prisma)
    assert "project-x" in result
    # project-x has no direct tag but expands to both children's models
    assert sorted(result["project-x"].model_names) == ["dall-e-3", "gpt-4"]
    assert result["project-x"].deployment_count == 0  # no direct deployments
    assert sorted(result["project-x"].child_groups) == ["image", "reasoning"]
    # Children expose their parent
    assert result["image"].parent_groups == ["project-x"]
    assert result["reasoning"].parent_groups == ["project-x"]


@pytest.mark.asyncio
async def test_get_all_access_groups_surfaces_pure_composition_groups():
    """A group that exists only in the membership table (no deployment tag)
    must still appear in the response so existence checks see it."""
    prisma = _make_prisma(
        deployment_rows=[_deployment("m1", "dall-e-3", ["image"])],
        membership_rows=[_row("composite", "image")],
    )
    result = await get_all_access_groups_from_db(prisma_client=prisma)
    assert "composite" in result
    assert result["composite"].deployment_count == 0
    assert result["composite"].child_groups == ["image"]
    assert result["composite"].model_names == ["dall-e-3"]
