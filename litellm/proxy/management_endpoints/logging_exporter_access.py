"""Shared access predicate for admin-owned logging destinations.

``credential_info.access`` answers "who may see/assign this destination" — it is
visibility, decoupled from enablement (which lives in ``metadata.logging_exporters``
and the explicit ``auto_enable`` flag). The request-time resolver and the write-time
validator gate on the SAME predicate so "visible" means the same thing on both sides:
a team/org admin can only assign destinations they can see, and the resolver
defensively re-checks visibility at request time.
"""

from typing import Optional

AUTO_ENABLE_KEY = "auto_enable"


def access_grants(access: object, team_id: Optional[str], org_id: Optional[str]) -> bool:
    """Whether a destination's ``access`` makes it visible to this identity.

    ``global`` reaches everyone; otherwise the identity's team or org must be listed.
    A missing or malformed ``access`` grants no one (fail closed): visibility must be
    an explicit admin grant, never an accident of an absent field.
    """
    if not isinstance(access, dict):
        return False
    if access.get("global") is True:
        return True
    teams = access.get("teams")
    if team_id is not None and isinstance(teams, (list, tuple)) and team_id in teams:
        return True
    orgs = access.get("orgs")
    return org_id is not None and isinstance(orgs, (list, tuple)) and org_id in orgs


def is_auto_enable(credential_info: object) -> bool:
    """Whether a destination is an explicit global/default (auto-enabled everywhere).

    This is the deliberate replacement for the old behavior where ``access.global``
    implicitly auto-enabled a destination for every request. Enablement is now opt-in:
    only ``auto_enable`` turns a destination on without being named.
    """
    return isinstance(credential_info, dict) and credential_info.get(AUTO_ENABLE_KEY) is True
