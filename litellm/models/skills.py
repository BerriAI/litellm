"""
Skills table model.

Canonical definition for ``litellm_skillstable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_SkillsTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_SkillsTable record"""

    skill_id: str
    display_title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    source: str = "custom"
    latest_version: Optional[str] = None
    file_content: Optional[bytes] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
