"""Shared visibility predicate for admin-owned logging destinations.

``credential_info.access`` answers "who may see and assign this destination". It
is visibility, decoupled from enablement (a named assignment plus the explicit
``auto_enable`` default-on flag). One predicate serves three callers so "visible"
means the same thing everywhere: the ``GET /credentials`` list filter, the
assignment validator, and the request-time resolver. When these disagree a
non-admin can be shown, or route traffic to, a destination it should never see;
keeping the check in one place is what prevents that.

``access_grants`` is the primitive: does this ``access`` reach a caller whose
admin scope is the given set of team ids and org ids. Single-identity callers
(the resolver, the per-write assignment gate) pass a one-element scope built with
``identity_scope``; the list endpoint, whose caller may administer several teams
and orgs, passes the full scope.
"""

from typing import Optional

from pydantic import ValidationError

from litellm.models.credentials import CredentialAccess, CredentialInfo


def parse_credential_info(raw: object) -> Optional[CredentialInfo]:
    """Parse stored ``credential_info`` into the typed model, or ``None`` when it is
    absent or malformed.

    Callers fail closed on ``None``: a destination whose stored ``access`` cannot be
    parsed (a legacy shape the strict read model rejects) is treated as invisible
    rather than granted to everyone.
    """
    if not isinstance(raw, dict):
        return None
    try:
        return CredentialInfo.model_validate(raw)
    except ValidationError:
        return None


def identity_scope(team_id: Optional[str], org_id: Optional[str]) -> tuple[frozenset[str], frozenset[str]]:
    """A single request identity's admin scope as ``(team_ids, org_ids)`` for
    ``access_grants`` / ``is_destination_visible``."""
    return (
        frozenset({team_id}) if team_id else frozenset(),
        frozenset({org_id}) if org_id else frozenset(),
    )


def access_grants(
    access: Optional[CredentialAccess],
    team_ids: frozenset[str],
    org_ids: frozenset[str],
) -> bool:
    """Whether ``access`` makes a destination visible to a caller admin-scoped to
    ``team_ids`` / ``org_ids``.

    ``global`` reaches everyone; otherwise one of the caller's admin teams or orgs
    must be granted. A missing ``access`` grants no one (fail closed): visibility is
    an explicit admin grant, never the accident of an absent field.
    """
    if access is None:
        return False
    if access.global_:
        return True
    if not team_ids.isdisjoint(access.teams):
        return True
    return not org_ids.isdisjoint(access.orgs)


def is_destination_visible(
    info: CredentialInfo,
    team_ids: frozenset[str],
    org_ids: frozenset[str],
) -> bool:
    """Whether a caller admin-scoped to ``team_ids`` / ``org_ids`` may see and assign
    this destination: an auto-enabled default, or a grant that reaches their scope.
    """
    return info.auto_enable or access_grants(info.access, team_ids, org_ids)
