"""
Verification token table model.

Canonical definition for ``litellm_verificationtoken``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime

from pydantic import ConfigDict

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_VerificationToken(LiteLLMPydanticObjectBase):
    token: str | None = None
    key_name: str | None = None
    key_alias: str | None = None
    spend: float = 0.0
    max_budget: float | None = None
    expires: str | datetime | None = None
    models: list = []
    aliases: dict = {}
    config: dict = {}
    user_id: str | None = None
    team_id: str | None = None
    agent_id: str | None = None
    project_id: str | None = None
    max_parallel_requests: int | None = None
    metadata: dict = {}
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    allowed_cache_controls: list | None = []
    allowed_routes: list | None = []
    key_type: str | None = None
    permissions: dict = {}
    model_spend: dict = {}
    model_max_budget: dict = {}
    budget_fallbacks: dict[str, list[str]] = {}
    soft_budget_cooldown: bool = False
    blocked: bool | None = None
    litellm_budget_table: dict | None = None
    budget_id: str | None = None
    org_id: str | None = None  # org id for a given key
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
    settings_updated_at: datetime | None = None
    last_active: datetime | None = None
    object_permission_id: str | None = None
    object_permission: LiteLLM_ObjectPermissionTable | None = None
    access_group_ids: list[str] | None = None
    rotation_count: int | None = 0
    auto_rotate: bool | None = False
    rotation_interval: str | None = None
    last_rotation_at: datetime | None = None
    key_rotation_at: datetime | None = None
    router_settings: dict | None = None
    budget_limits: list[dict] | None = None
    model_config = ConfigDict(protected_namespaces=())


class LiteLLM_DeletedVerificationToken(LiteLLM_VerificationToken):
    """Audit record for deleted keys; mirrors the token plus deletion metadata."""

    id: str | None = None
    deleted_at: datetime | None = None
    deleted_by: str | None = None
    deleted_by_api_key: str | None = None
    litellm_changed_by: str | None = None

    model_config = ConfigDict(protected_namespaces=())
