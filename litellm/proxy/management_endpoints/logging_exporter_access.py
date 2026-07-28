"""Request-time routing predicate for admin-owned logging destinations.

``credential_info.access`` answers "which identities' traces may this destination
receive". It is routing scope, decoupled from enablement (a named assignment plus
the explicit ``auto_enable`` default-on flag). The request-time resolver in
``litellm_pre_call_utils`` is the consumer: at call time it checks whether the
request's team/org is granted before firing the destination.

``access_grants`` is the primitive: does this ``access`` reach an identity whose
scope is the given set of team ids and org ids. The resolver passes a
one-element scope built with ``identity_scope``.
"""

from collections.abc import Sequence

from pydantic import ValidationError

import litellm
from litellm.models.credentials import CredentialAccess, CredentialInfo


def parse_credential_info(raw: object) -> CredentialInfo | None:
    """Parse stored ``credential_info`` into the typed model, or ``None`` when it is
    absent or malformed.

    Callers fail closed on ``None``: a destination whose stored ``access`` cannot be
    parsed (a legacy shape the strict read model rejects) is treated as granted to
    no one rather than granted to everyone.
    """
    if not isinstance(raw, dict):
        return None
    try:
        return CredentialInfo.model_validate(raw)
    except ValidationError:
        return None


def identity_scope(team_id: str | None, org_id: str | None) -> tuple[frozenset[str], frozenset[str]]:
    """A single request identity's scope as ``(team_ids, org_ids)`` for
    ``access_grants``."""
    return (
        frozenset({team_id}) if team_id else frozenset(),
        frozenset({org_id}) if org_id else frozenset(),
    )


def resolved_logging_exporter_names(
    assigned: Sequence[str] | None,
    team_id: str | None,
    org_id: str | None,
) -> tuple[str, ...]:
    """Destination names that will receive this identity's traces, for disclosure on
    the team/org info pages.

    Mirrors the request-time resolver's selection: a logging destination is included
    when its ``access`` grants the identity AND it is either ``auto_enable`` or named
    in ``assigned`` (the identity's own ``logging_exporters``). Names only; endpoints,
    headers, and the access map itself stay proxy-admin information.
    """
    team_ids, org_ids = identity_scope(team_id, org_id)
    own = frozenset(str(name) for name in (assigned or ()))
    selected = tuple(
        credential.credential_name
        for credential in litellm.credential_list
        if (info := parse_credential_info(credential.credential_info)) is not None
        and info.credential_type == "logging"
        and access_grants(info.access, team_ids, org_ids)
        and (info.auto_enable or credential.credential_name in own)
    )
    return tuple(dict.fromkeys(selected))


def access_grants(
    access: CredentialAccess | None,
    team_ids: frozenset[str],
    org_ids: frozenset[str],
) -> bool:
    """Whether ``access`` grants a destination to an identity scoped to
    ``team_ids`` / ``org_ids``.

    ``global`` reaches everyone; otherwise one of the identity's teams or orgs
    must be granted. A missing ``access`` grants no one (fail closed): routing is
    an explicit admin grant, never the accident of an absent field.
    """
    if access is None:
        return False
    if access.global_:
        return True
    if not team_ids.isdisjoint(access.teams):
        return True
    return not org_ids.isdisjoint(access.orgs)
