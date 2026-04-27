from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class BulkUpdateKeyRequestItem(BaseModel):
    """Individual key update request item"""

    key: str  # Key identifier (token)
    budget_id: Optional[str] = None  # Budget ID associated with the key
    max_budget: Optional[float] = None  # Max budget for key
    team_id: Optional[str] = None  # Team ID associated with key
    tags: Optional[List[str]] = None  # Tags for organizing keys


class BulkUpdateKeyRequest(BaseModel):
    """Request for bulk key updates"""

    keys: List[BulkUpdateKeyRequestItem]


class SuccessfulKeyUpdate(BaseModel):
    """Successfully updated key with its updated information"""

    key: str
    key_info: Dict[str, Any]


class FailedKeyUpdate(BaseModel):
    """Failed key update with reason"""

    key: str
    key_info: Optional[Dict[str, Any]] = None
    failed_reason: str


class BulkUpdateKeyResponse(BaseModel):
    """Response for bulk key update operations"""

    total_requested: int
    successful_updates: List[SuccessfulKeyUpdate]
    failed_updates: List[FailedKeyUpdate]


class KeyUpdateFields(BaseModel):
    """Allowlist of bulk-broadcastable fields for /team/key/bulk_update; `extra="forbid"` blocks RBAC/ownership/scope mutations even by team admins."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Budgets
    max_budget: Optional[float] = None
    budget_id: Optional[str] = None
    budget_duration: Optional[str] = None
    budget_limits: Optional[List[Any]] = None
    model_max_budget: Optional[Dict[str, Any]] = None

    # Rate limits
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    model_tpm_limit: Optional[Dict[str, Any]] = None
    model_rpm_limit: Optional[Dict[str, Any]] = None
    max_parallel_requests: Optional[int] = None
    rpm_limit_type: Optional[
        Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"]
    ] = None
    tpm_limit_type: Optional[
        Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"]
    ] = None

    # Temporary budget grants (auto-expire). `spend` deliberately omitted — bulk-zeroing it bypasses budget enforcement; admin-only via /key/update.
    temp_budget_increase: Optional[float] = None
    temp_budget_expiry: Optional[datetime] = None

    # Expiry
    duration: Optional[str] = None

    # Operational metadata
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_temp_budget(self) -> "KeyUpdateFields":
        if self.temp_budget_increase is not None or self.temp_budget_expiry is not None:
            if self.temp_budget_increase is None or self.temp_budget_expiry is None:
                raise ValueError(
                    "temp_budget_increase and temp_budget_expiry must be set together"
                )
        return self

    @model_validator(mode="after")
    def require_at_least_one_field(self) -> "KeyUpdateFields":
        # Reject empty payload — would iterate every key with no-op writes.
        if not self.model_fields_set:
            raise ValueError("update_fields must specify at least one field to update.")
        return self


class BulkUpdateTeamKeysRequest(BaseModel):
    """Apply one update payload to many keys inside a team; provide either `key_ids` or `all_keys_in_team=True`."""

    team_id: str
    key_ids: Optional[List[str]] = None
    all_keys_in_team: bool = False
    update_fields: KeyUpdateFields

    @model_validator(mode="after")
    def validate_selection(self) -> "BulkUpdateTeamKeysRequest":
        has_key_ids = self.key_ids is not None and len(self.key_ids) > 0
        if has_key_ids and self.all_keys_in_team:
            raise ValueError(
                "Provide either `key_ids` or `all_keys_in_team=True`, not both."
            )
        if not has_key_ids and not self.all_keys_in_team:
            raise ValueError(
                "Must provide either `key_ids` (non-empty) or `all_keys_in_team=True`."
            )
        return self
