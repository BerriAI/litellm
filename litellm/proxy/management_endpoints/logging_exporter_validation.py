"""Shape validation for an admin-owned logging destination's ``credential_info.access``.

Which identities a destination fires for is governed entirely by its
``credential_info.access``; the resolver (``litellm_pre_call_utils``) evaluates that at
request time. This module only checks that a write sets a well-formed ``access`` object.
"""

from collections.abc import Mapping

from fastapi import HTTPException, status


def validate_credential_access(credential_info: Mapping[str, object] | None) -> None:
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
