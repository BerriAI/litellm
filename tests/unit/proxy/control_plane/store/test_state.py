from dataclasses import replace
from typing import Final

import pytest

from litellm.proxy.control_plane.auth.rbac import Scope
from litellm.proxy.control_plane.store.models import Change, Entry
from tests.unit.proxy.control_plane.store.support import container, role_entry


def test_replayed_out_of_order_and_deleted_entries_cannot_restore_stale_permissions() -> None:
    rig: Final = container()
    state: Final = rig.runtime_config.state
    state.apply(role_entry("10-0", "runtime", "fleet:read"))
    state.apply(role_entry("9-9", "runtime", "roles:write"))
    assert rig.policy.scopes_for("runtime") == frozenset({Scope.FLEET_READ})
    assert not state.apply(role_entry("10-0", "runtime", "roles:write"))
    assert rig.policy.scopes_for("runtime") == frozenset({Scope.FLEET_READ})
    deleted: Final = Entry(revision="11-0", change=Change(op="delete", kind="role", name="runtime", actor="operator"))
    assert state.apply(deleted)
    assert "runtime" not in rig.policy.names
    state.reload(())
    assert not state.apply(role_entry("10-1", "runtime", "roles:write"))
    assert not state.apply(deleted)
    assert "runtime" not in rig.policy.names
    assert state.apply(role_entry("12-0", "runtime", "usage:read"))
    assert rig.policy.scopes_for("runtime") == frozenset({Scope.USAGE_READ})


def test_config_entries_and_default_roles_cannot_be_replaced_by_runtime_rows() -> None:
    rig: Final = container()
    before: Final = rig.policy.scopes_for("admin")
    rig.runtime_config.state.apply(role_entry("1-0", "admin"))
    assert rig.policy.scopes_for("admin") == before


def test_invalid_parent_does_not_replace_the_live_snapshot() -> None:
    rig: Final = container()
    state: Final = rig.runtime_config.state
    original: Final = state.view
    assert not state.apply(
        Entry(
            revision="1-0",
            change=Change(
                op="upsert", kind="role", name="bad", entity={"scopes": [], "inherits": ["absent"]}, actor="operator"
            ),
        )
    )
    assert state.view is original
