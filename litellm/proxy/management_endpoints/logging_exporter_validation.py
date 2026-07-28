"""Validation for admin-owned logging-exporter assignment on key/team/org.

An identity's ``metadata.logging_exporters`` binds it to admin-owned trace
destinations. Only the proxy admin may write it, and every name must be a registered
logging credential. Which identities a destination actually fires for is governed by
the destination's own ``credential_info.access``; the resolver
(``litellm_pre_call_utils``) evaluates that at request time.
"""

from fastapi import HTTPException, status

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

LOGGING_EXPORTERS_KEY = "logging_exporters"


def is_admin_gated_credential_info(credential_info: dict | None) -> bool:
    """Whether a credential write must be proxy-admin only.

    True when the credential is a logging destination or carries an ``access`` grant,
    since both control where other tenants' traces are exported.
    """
    if not isinstance(credential_info, dict):
        return False
    return credential_info.get("credential_type") == "logging" or "access" in credential_info


def validate_credential_access(credential_info: dict | None) -> None:
    """Validate ``credential_info.access`` shape when the write sets one.

    No-op when ``access`` is absent. Otherwise it must be an object whose ``global`` (if
    present) is a bool and whose ``teams``/``orgs`` (if present) are lists of strings.
    Per-key access is intentionally unsupported on a destination.
    """
    if not isinstance(credential_info, dict) or "access" not in credential_info:
        return
    access = credential_info["access"]
    if not isinstance(access, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "credential_info.access must be an object"},
        )
    if "global" in access and not isinstance(access["global"], bool):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "access.global must be a boolean"},
        )
    for field in ("teams", "orgs"):
        bucket = access.get(field)
        if bucket is not None and not (isinstance(bucket, list) and all(isinstance(item, str) for item in bucket)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"access.{field} must be a list of strings"},
            )
    unknown = set(access) - {"global", "teams", "orgs"}
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"access contains unknown field(s): {sorted(unknown)}"},
        )


def _logging_credentials_by_name() -> dict[str, dict]:
    return {
        credential.credential_name: (credential.credential_info or {})
        for credential in litellm.credential_list
        if (credential.credential_info or {}).get("credential_type") == "logging"
    }


def _logging_credential_names() -> set[str]:
    return set(_logging_credentials_by_name())


def _validate_exporters_shape_and_names(exporters: object) -> None:
    """Common shape + registry check shared by every entry point."""
    if not isinstance(exporters, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "logging_exporters must be a list of credential names"},
        )
    known = _logging_credential_names()
    unknown = [name for name in exporters if name not in known]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": (
                    f"Unknown or non-logging credential(s): {unknown}. Register them "
                    "as logging credentials before assigning."
                )
            },
        )


def _exporter_value_changes(
    requested_metadata: dict | None,
    existing_metadata: dict | None,
) -> bool:
    """True if the effective ``metadata.logging_exporters`` value would change.

    An update endpoint that REPLACES stored metadata with ``requested_metadata``
    will drop ``logging_exporters`` when the new payload omits it. So a write
    requires authorization whenever:

    - the new metadata sets ``logging_exporters`` (the previously-handled case), OR
    - the new metadata is provided but omits ``logging_exporters`` while the
      stored metadata had one (removal-via-omission, Veria F4).

    Returns False when stored and requested values match exactly, or when the
    update doesn't touch metadata at all.
    """
    if not isinstance(requested_metadata, dict):
        return False
    new_has = LOGGING_EXPORTERS_KEY in requested_metadata
    existing = existing_metadata.get(LOGGING_EXPORTERS_KEY) if isinstance(existing_metadata, dict) else None
    existing_has = existing is not None
    if not new_has and not existing_has:
        return False
    if new_has and not existing_has:
        return True
    if not new_has and existing_has:
        return True
    return requested_metadata.get(LOGGING_EXPORTERS_KEY) != existing


def validate_logging_exporter_field(
    requested_exporters: list | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    existing_exporters: list | None = None,
) -> None:
    """Authorize a typed ``logging_exporters`` write (proxy-admin only).

    Adapts the typed list to the metadata-shaped input the shared assignment
    validator expects, so the authorization logic lives in one place.
    ``requested_exporters is None`` means the field was not provided (no-op); an
    empty list is an explicit clear and is gated like any other change.
    ``existing_exporters`` is the stored column value, passed so a change is
    detected and a non-admin cannot silently clear an admin-assigned value.
    """
    requested_metadata = None if requested_exporters is None else {LOGGING_EXPORTERS_KEY: requested_exporters}
    existing_metadata = None if existing_exporters is None else {LOGGING_EXPORTERS_KEY: existing_exporters}
    validate_logging_exporter_assignment(
        requested_metadata,
        user_api_key_dict,
        existing_metadata=existing_metadata,
    )


def validate_logging_exporter_assignment(
    metadata: dict | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    existing_metadata: dict | None = None,
) -> None:
    """Validate a ``metadata.logging_exporters`` write on key / team / org endpoints.

    Proxy-admin only. No-op when the update does not change the effective
    ``logging_exporters`` value; otherwise a non-proxy-admin is rejected.

    Update paths replace stored metadata wholesale, so a caller could drop an
    admin-assigned exporter by sending ``metadata`` without ``logging_exporters``.
    Pass ``existing_metadata`` from the loaded row so removal-via-omission is gated
    too (Veria F4). Every exporter name (when present) must resolve to a registered
    logging credential.
    """
    if not _exporter_value_changes(metadata, existing_metadata):
        return
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Only the proxy admin can assign logging exporters"},
        )
    requested = metadata.get(LOGGING_EXPORTERS_KEY) if isinstance(metadata, dict) else None
    if requested is not None:
        _validate_exporters_shape_and_names(requested)
