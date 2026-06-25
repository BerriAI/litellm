"""Pure tagged-union decision for who can patch a logging-credential.

A logging credential controls where other tenants' traces are exported, so
``credential_info.access`` is the only field a non-admin caller may touch, and
only to add their own team_id(s) to ``access.teams``. Everything else
(values, host, type, description, ``global``, ``orgs``, foreign team_ids,
or removing existing grants) stays proxy-admin only.

The decision is a value (Allow vs Deny(reason)), kept separate from the
endpoint so it can be unit-tested exhaustively without spinning up FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, cast


@dataclass(frozen=True, slots=True)
class Allow:
    tag: Literal["allow"] = "allow"


@dataclass(frozen=True, slots=True)
class Deny:
    reason: str
    tag: Literal["deny"] = "deny"


Decision = Allow | Deny


_IMMUTABLE_INFO_FIELDS = frozenset(
    {"credential_type", "description", "host", "endpoint"}
)


_EMPTY: Mapping[str, object] = {}


def _access(info: Mapping[str, object] | None) -> Mapping[str, object]:
    if not isinstance(info, Mapping):
        return _EMPTY
    raw = info.get("access")
    if isinstance(raw, Mapping):
        return cast(Mapping[str, object], raw)
    return _EMPTY


def _teams_set(access: Mapping[str, object]) -> frozenset[str]:
    raw = access.get("teams")
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    teams: tuple[object, ...] = tuple(cast(tuple[object, ...], raw))
    return frozenset(item for item in teams if isinstance(item, str))


def decide_credential_patch(
    *,
    is_proxy_admin: bool,
    caller_team_admin_ids: frozenset[str],
    existing_info: Mapping[str, object] | None,
    patch_info: Mapping[str, object] | None,
    patch_values: Mapping[str, object] | None,
    patch_name_changed: bool,
) -> Decision:
    """Return Allow or Deny(reason) for a PATCH /credentials/{name} request.

    Proxy admins always pass. A team admin only passes when the patch (a) does
    not change ``credential_values`` or ``credential_name``, (b) does not
    modify any immutable ``credential_info`` field, and (c) limits its
    ``access`` change to appending team_ids the caller is team-admin of to
    ``access.teams`` (no removals, no foreign ids, no ``global``/``orgs``
    edits).
    """
    if is_proxy_admin:
        return Allow()

    if not caller_team_admin_ids:
        return Deny("Only the proxy admin can manage logging credentials")

    if patch_name_changed:
        return Deny("credential_name is proxy-admin only")

    if patch_values:
        return Deny("credential_values is proxy-admin only")

    if not isinstance(patch_info, Mapping):
        # Nothing to do, but a team-admin should only call PATCH to change
        # access. An empty/None info is a no-op and we refuse it loudly so
        # the caller corrects the request shape.
        return Deny("patch must set credential_info.access for team-admin writes")

    touched_fields = frozenset(patch_info.keys())
    forbidden = touched_fields & _IMMUTABLE_INFO_FIELDS
    if forbidden:
        return Deny(
            "credential_info fields are proxy-admin only: "
            + ", ".join(sorted(forbidden))
        )

    if touched_fields - {"access"}:
        return Deny(
            "team-admin may only patch credential_info.access; got: "
            + ", ".join(sorted(touched_fields))
        )

    patch_access = _access(patch_info)
    if not patch_access:
        return Deny("credential_info.access must be set for team-admin writes")

    if "global" in patch_access:
        return Deny("access.global is proxy-admin only")
    if "orgs" in patch_access:
        return Deny("access.orgs is proxy-admin only")

    existing_access = _access(existing_info)
    existing_teams = _teams_set(existing_access)
    patch_teams = _teams_set(patch_access)

    removed = existing_teams - patch_teams
    foreign_removed = removed - caller_team_admin_ids
    if foreign_removed:
        return Deny(
            "team-admin may only revoke their own team grants; foreign team_ids: "
            + ", ".join(sorted(foreign_removed))
        )

    added = patch_teams - existing_teams
    foreign_added = added - caller_team_admin_ids
    if foreign_added:
        return Deny(
            "team-admin may only grant their own team_ids: "
            + ", ".join(sorted(foreign_added))
        )

    return Allow()
