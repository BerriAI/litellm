"""Request-time routing predicate for admin-owned logging destinations.

``credential_info.access`` answers "which identities' traces may this destination
receive". It is the sole routing determinant: at call time the resolver in
``litellm_pre_call_utils`` fires a destination for a request exactly when the
request's team/org is granted by that destination's ``access``.

``access_grants`` is the primitive: does this ``access`` reach an identity whose
scope is the given set of team ids and org ids. The resolver passes a
one-element scope built with ``identity_scope``.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.integrations.otel.model.config import is_otel_v2_enabled
from litellm.models.credentials import CredentialAccess, CredentialInfo, CredentialItem

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination


class _LoggingDestinationTag(BaseModel):
    """Lenient read of just the ``credential_type`` tag, ignoring the rest of
    ``credential_info`` (including a possibly-malformed ``access``)."""

    model_config = ConfigDict(extra="ignore")
    credential_type: str | None = None


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


def is_logging_credential(raw: object) -> bool:
    """Whether ``credential_info`` is tagged as an admin-owned logging destination.

    The ``access`` shape validation and the ``credential_info`` subfield merge are
    scoped to these; a provider credential is left on its base replace-and-accept path.

    This keys off the ``credential_type`` tag alone and does not parse ``access``: a
    destination carrying a malformed ``access`` is still a logging destination, and the
    point of the gate is to route it into ``validate_credential_access`` so that bad
    ``access`` is rejected rather than stored.
    """
    try:
        return _LoggingDestinationTag.model_validate(raw).credential_type == "logging"
    except ValidationError:
        return False


def identity_scope(team_id: str | None, org_id: str | None) -> tuple[frozenset[str], frozenset[str]]:
    """A single request identity's scope as ``(team_ids, org_ids)`` for
    ``access_grants``."""
    return (
        frozenset({team_id}) if team_id else frozenset(),
        frozenset({org_id}) if org_id else frozenset(),
    )


def resolved_logging_exporter_names(
    team_id: str | None,
    org_id: str | None,
) -> tuple[str, ...]:
    """Destination names that will receive this identity's traces, for disclosure on
    the team/org info pages.

    Mirrors the request-time resolver's selection so it never advertises an exporter that
    receives no traces: a destination is disclosed only when its ``access`` grants the
    identity AND it actually builds. Names only; endpoints, headers, and the access map
    itself stay proxy-admin information.

    The resolver also dedupes destinations that resolve to the same endpoint, headers and
    resource attributes, so this drops the later duplicates too; listing both names would
    advertise a second exporter that no trace ever reaches.

    Gated on ``is_otel_v2_enabled`` for parity with the resolver, which returns nothing
    when the flag is off: disclosing a destination the request path would never fire
    would claim traces are exported when none are.
    """
    if not is_otel_v2_enabled():
        return ()
    team_ids, org_ids = identity_scope(team_id, org_id)
    granted = tuple(
        (credential.credential_name, _export_target(built[1]))
        for credential in litellm.credential_list
        if (info := parse_credential_info(credential.credential_info)) is not None
        and info.credential_type == "logging"
        and access_grants(info.access, team_ids, org_ids)
        if (built := destination_for_credential(credential)) is not None
    )
    targets = tuple(target for _name, target in granted)
    return tuple(name for index, (name, target) in enumerate(granted) if target not in targets[:index])


def _export_target(destination: "OtelDestination") -> tuple[object, ...]:
    """What two destinations must share to be the same export target.

    Two names for one target would advertise a second exporter that no trace ever reaches,
    so the disclosure keeps only the first credential that resolves to each of these.
    """
    return (
        destination.endpoint,
        tuple(sorted(destination.headers.items())),
        tuple(sorted(destination.resource_attributes.items())),
    )


def destination_for_credential(credential: CredentialItem) -> 'tuple[str, "OtelDestination"] | None':
    """The ``(backend, destination)`` this logging credential resolves to, or ``None`` when it
    resolves to nothing.

    A credential builds only when it names a backend (``credential_info.description``) and
    ``build_destination`` accepts its values. Shared by the request-time resolver (which fans
    out to the built destinations) and the team/org disclosure (which must not advertise a
    destination that resolves to nothing), so the two cannot drift apart.
    """
    from litellm.integrations.otel.presets.destinations import build_destination

    backend = (credential.credential_info or {}).get("description")
    if not backend:
        return None
    # Drop unset (``None``) values rather than stringifying them: ``str(None)`` is the
    # literal ``"None"``, which would land in the exporter endpoint/headers and break
    # the export (e.g. an empty ``otel_endpoint`` becoming the URL ``"None"``).
    values = {str(key): str(value) for key, value in (credential.credential_values or {}).items() if value is not None}
    destination = build_destination(backend, values)
    return None if destination is None else (backend, destination)


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
