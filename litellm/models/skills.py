"""
Skills table model.

Canonical definition for ``litellm_skillstable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Any

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_SkillsTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_SkillsTable record"""

    skill_id: str
    display_title: str | None = None
    description: str | None = None
    instructions: str | None = None
    source: str = "custom"
    latest_version: str | None = None
    file_content: bytes | None = None
    file_name: str | None = None
    file_type: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class NewSkillRequest(LiteLLMPydanticObjectBase):
    """Request to create a new skill in LiteLLM database"""

    display_title: str | None = None
    description: str | None = None
    instructions: str | None = None
    file_content: bytes | None = None  # Binary content of skill files (zip)
    file_name: str | None = None  # Original filename
    file_type: str | None = None  # MIME type (e.g., "application/zip")
    metadata: dict[str, Any] | None = None
    authorization_url: str | None = None
    token_url: str | None = None
    registration_url: str | None = None
