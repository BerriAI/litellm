"""Validation for admin-owned logging-exporter assignment on key/team/org.

An identity's ``metadata.logging_exporters`` binds it to admin-owned trace
destinations. Assigning is proxy-admin only, and every name must be a registered
logging credential, so a key/team/org can only point at destinations the admin has
provisioned. The resolver (``litellm_pre_call_utils``) trusts this at request time.
"""

from typing import Optional

from fastapi import HTTPException, status

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

LOGGING_EXPORTERS_KEY = "logging_exporters"


def is_admin_gated_credential_info(credential_info: Optional[dict]) -> bool:
    """Whether a credential write must be proxy-admin only.

    True when the credential is a logging destination or carries an ``access`` grant,
    since both control where other tenants' traces are exported.
    """
    if not isinstance(credential_info, dict):
        return False
    return (
        credential_info.get("credential_type") == "logging"
        or "access" in credential_info
    )


def validate_credential_access(credential_info: Optional[dict]) -> None:
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
        if bucket is not None and not (
            isinstance(bucket, list) and all(isinstance(item, str) for item in bucket)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"access.{field} must be a list of strings"},
            )


def _logging_credential_names() -> set:
    return {
        credential.credential_name
        for credential in litellm.credential_list
        if (credential.credential_info or {}).get("credential_type") == "logging"
    }


def _validate_exporters_shape_and_names(exporters: object) -> list[str]:
    """Common shape + registry check shared by the two validator entry points."""
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
    return [name for name in exporters if isinstance(name, str)]


def validate_logging_exporter_assignment(
    metadata: Optional[dict], user_api_key_dict: UserAPIKeyAuth
) -> None:
    """Validate a ``metadata.logging_exporters`` assignment for proxy-admin-only paths.

    Used by key/org writes and team creation, where the team being affected has
    no pre-existing team-admin to delegate to. ``/team/update`` uses
    ``validate_team_logging_exporter_assignment`` instead.
    """
    if not isinstance(metadata, dict) or LOGGING_EXPORTERS_KEY not in metadata:
        return
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Only the proxy admin can assign logging exporters"},
        )
    _validate_exporters_shape_and_names(metadata.get(LOGGING_EXPORTERS_KEY))


def validate_team_logging_exporter_assignment(
    metadata: Optional[dict],
    user_api_key_dict: UserAPIKeyAuth,
    is_team_admin: bool,
) -> None:
    """``metadata.logging_exporters`` validator for ``/team/update`` only.

    Proxy admins and team-admins of the team being edited may write the field;
    every name still has to resolve to a registered logging credential.
    """
    if not isinstance(metadata, dict) or LOGGING_EXPORTERS_KEY not in metadata:
        return
    is_proxy_admin = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
    if not (is_proxy_admin or is_team_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    "Only the proxy admin or a team admin of this team can assign "
                    "logging exporters"
                )
            },
        )
    _validate_exporters_shape_and_names(metadata.get(LOGGING_EXPORTERS_KEY))
