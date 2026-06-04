"""
Credentials domain model.
"""

from typing import Any, Dict, Optional

from litellm.models.base import DomainModel


class Credentials(DomainModel):
    """Domain model for credentials storage."""

    credential_id: Optional[str] = None
    credential_name: str
    credential_values: Dict[str, Any]
    credential_info: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
