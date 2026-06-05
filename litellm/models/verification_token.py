"""
Verification token table model.

Canonical definition for ``litellm_verificationtoken``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import ConfigDict

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_VerificationToken(LiteLLMPydanticObjectBase):
    token: Optional[str] = None
    key_name: Optional[str] = None
    key_alias: Optional[str] = None
    spend: float = 0.0
    max_budget: Optional[float] = None
    expires: Optional[Union[str, datetime]] = None
    models: List = []
    aliases: Dict = {}
    config: Dict = {}
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    agent_id: Optional[str] = None
    project_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    metadata: Dict = {}
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    allowed_cache_controls: Optional[list] = []
    allowed_routes: Optional[list] = []
    permissions: Dict = {}
    model_spend: Dict = {}
    model_max_budget: Dict = {}
    soft_budget_cooldown: bool = False
    blocked: Optional[bool] = None
    litellm_budget_table: Optional[dict] = None
    budget_id: Optional[str] = None
    org_id: Optional[str] = None  # org id for a given key
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    last_active: Optional[datetime] = None
    object_permission_id: Optional[str] = None
    object_permission: Optional[LiteLLM_ObjectPermissionTable] = None
    access_group_ids: Optional[List[str]] = None
    rotation_count: Optional[int] = 0
    auto_rotate: Optional[bool] = False
    rotation_interval: Optional[str] = None
    last_rotation_at: Optional[datetime] = None
    key_rotation_at: Optional[datetime] = None
    router_settings: Optional[dict] = None
    budget_limits: Optional[List[dict]] = None
    model_config = ConfigDict(protected_namespaces=())


class LiteLLM_DeletedVerificationToken(LiteLLM_VerificationToken):
    """Audit record for deleted keys; mirrors the token plus deletion metadata."""

    id: Optional[str] = None
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    deleted_by_api_key: Optional[str] = None
    litellm_changed_by: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())
