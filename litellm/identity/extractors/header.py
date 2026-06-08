"""Header-driven extraction of non-credential identity fields."""

from typing import Optional

AUDIT_CHANGED_BY_HEADER = "litellm-changed-by"


def extract_audit_changed_by(headers: Optional[dict]) -> Optional[str]:
    """Read the ``litellm-changed-by`` header used for management-API audit."""
    if not headers:
        return None

    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == AUDIT_CHANGED_BY_HEADER:
            if isinstance(value, str) and value:
                return value
    return None
