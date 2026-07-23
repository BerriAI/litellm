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
from typing import Literal, Mapping

from litellm.models.credentials import CredentialAccess, CredentialInfo

OPAQUE_DENY_REASON = "Only the proxy admin can manage logging credentials"


@dataclass(frozen=True, slots=True)
class Allow:
    tag: Literal["allow"] = "allow"


@dataclass(frozen=True, slots=True)
class Deny:
    reason: str
    # When True the reason is derived from caller input (e.g. they typed a
    # foreign team_id) and is safe to surface. When False the reason would
    # confirm the stored credential is a logging destination; the endpoint
    # collapses these to OPAQUE_DENY_REASON so PATCH /credentials/{name}
    # can't be used as an existence oracle by a non-admin caller.
    from_user_input: bool = False
    tag: Literal["deny"] = "deny"


Decision = Allow | Deny


_IMMUTABLE_INFO_FIELDS = frozenset({"credential_type", "description", "host", "endpoint"})


def _patched_fields(info: CredentialInfo | None) -> frozenset[str]:
    """Names of credential_info fields the caller actually set in their patch."""
    if info is None:
        return frozenset()
    return frozenset(info.model_fields_set) | frozenset(info.model_extra.keys() if info.model_extra else ())


def _access_teams(access: CredentialAccess | None) -> frozenset[str]:
    return frozenset(access.teams) if access is not None else frozenset()


def decide_credential_patch(
    *,
    is_proxy_admin: bool,
    caller_team_admin_ids: frozenset[str],
    existing_info: CredentialInfo | None,
    patch_info: CredentialInfo | None,
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
        return Deny(OPAQUE_DENY_REASON)

    if patch_name_changed:
        return Deny("credential_name is proxy-admin only")

    if patch_values:
        return Deny("credential_values is proxy-admin only")

    touched = _patched_fields(patch_info)
    if not touched:
        return Deny("patch must set credential_info.access for team-admin writes")

    forbidden = touched & _IMMUTABLE_INFO_FIELDS
    if forbidden:
        return Deny("credential_info fields are proxy-admin only: " + ", ".join(sorted(forbidden)))

    if touched - {"access"}:
        return Deny("team-admin may only patch credential_info.access; got: " + ", ".join(sorted(touched)))

    assert patch_info is not None
    patch_access = patch_info.access
    if patch_access is None:
        return Deny("credential_info.access must be set for team-admin writes")

    # Touching global/orgs is allowed when the value matches the stored state
    # (the UI's edit modal sends the full access object back so unchecking
    # revokes; a no-op resend of global=false / orgs=[] must not be rejected).
    # Only block when the caller would actually CHANGE these.
    existing_access = existing_info.access if existing_info is not None else None
    access_touched = frozenset(patch_access.model_fields_set)
    existing_global = existing_access.global_ if existing_access is not None else False
    existing_orgs = frozenset(existing_access.orgs) if existing_access is not None else frozenset()
    if "global_" in access_touched and patch_access.global_ != existing_global:
        return Deny("access.global is proxy-admin only")
    if "orgs" in access_touched and frozenset(patch_access.orgs) != existing_orgs:
        return Deny("access.orgs is proxy-admin only")

    existing_teams = _access_teams(existing_access)
    patch_teams = _access_teams(patch_access)

    foreign_removed = (existing_teams - patch_teams) - caller_team_admin_ids
    if foreign_removed:
        # Do NOT echo the foreign team_ids -- they're stored values the
        # caller didn't send, so naming them would leak access list members.
        return Deny("team-admin may only revoke their own team grants")

    foreign_added = (patch_teams - existing_teams) - caller_team_admin_ids
    if foreign_added:
        return Deny(
            "team-admin may only grant their own team_ids: " + ", ".join(sorted(foreign_added)),
            from_user_input=True,
        )

    return Allow()
