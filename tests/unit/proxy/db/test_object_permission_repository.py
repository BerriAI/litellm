from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.repositories.object_permission_repository import ObjectPermissionRepository


class _RecordingPermissionTable:
    def __init__(self, stored: Mapping[str, object]) -> None:
        self.stored: Final = stored
        self.created: Final[list[Mapping[str, object]]] = []
        self.updated: Final[list[tuple[Mapping[str, object], Mapping[str, object]]]] = []

    async def create(self, data: Mapping[str, object]) -> Mapping[str, object]:
        self.created.append(data)
        return {**self.stored, **data}

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Mapping[str, object]:
        self.updated.append((where, data))
        return {**self.stored, **data}


def _repository(table: _RecordingPermissionTable) -> ObjectPermissionRepository:
    return ObjectPermissionRepository(SimpleNamespace(db=SimpleNamespace(litellm_objectpermissiontable=table)))


@pytest.mark.asyncio
async def test_create_permission_writes_only_the_fields_it_was_given() -> None:
    table: Final = _RecordingPermissionTable({"object_permission_id": "perm-1"})

    permission: Final = await _repository(table).create_permission(
        mcp_servers=["server-1"], mcp_tool_permissions={"server-1": ["search"]}, models=[]
    )

    assert table.created == [
        {"mcp_servers": ["server-1"], "mcp_tool_permissions": {"server-1": ["search"]}, "models": []}
    ]
    assert permission.model_dump(exclude_unset=True) == {
        "object_permission_id": "perm-1",
        "mcp_servers": ["server-1"],
        "mcp_tool_permissions": {"server-1": ["search"]},
        "models": [],
    }


@pytest.mark.asyncio
async def test_create_permission_without_fields_writes_an_empty_row() -> None:
    table: Final = _RecordingPermissionTable({"object_permission_id": "perm-1"})

    permission: Final = await _repository(table).create_permission()

    assert table.created == [{}]
    assert permission.model_dump(exclude_unset=True) == {"object_permission_id": "perm-1"}


@pytest.mark.asyncio
async def test_update_permission_changes_only_the_fields_it_was_given() -> None:
    table: Final = _RecordingPermissionTable(
        {"object_permission_id": "perm-1", "models": ["gpt-4o"], "agents": ["agent-1"]}
    )

    permission: Final = await _repository(table).update_permission("perm-1", models=["gpt-4o-mini"], skills=[])

    assert table.updated == [({"object_permission_id": "perm-1"}, {"models": ["gpt-4o-mini"], "skills": []})]
    assert permission is not None
    assert permission.model_dump(exclude_unset=True) == {
        "object_permission_id": "perm-1",
        "models": ["gpt-4o-mini"],
        "agents": ["agent-1"],
        "skills": [],
    }
