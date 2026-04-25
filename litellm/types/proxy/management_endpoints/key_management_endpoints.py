from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator

from litellm.proxy._types import KeyRequestBase


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


class KeyUpdateFields(KeyRequestBase):
    """
    Mirror of UpdateKeyRequest minus per-key identifiers (`key`, `key_alias`)
    and the scope guard (`team_id`). Used as the broadcast payload in
    BulkUpdateTeamKeysRequest — one set of fields applied to many keys.
    """

    duration: Optional[str] = None
    spend: Optional[float] = None
    metadata: Optional[dict] = None
    temp_budget_increase: Optional[float] = None
    temp_budget_expiry: Optional[datetime] = None
    auto_rotate: Optional[bool] = None
    rotation_interval: Optional[str] = None
    organization_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_temp_budget(self) -> "KeyUpdateFields":
        if self.temp_budget_increase is not None or self.temp_budget_expiry is not None:
            if self.temp_budget_increase is None or self.temp_budget_expiry is None:
                raise ValueError(
                    "temp_budget_increase and temp_budget_expiry must be set together"
                )
        return self

    @model_validator(mode="after")
    def reject_per_key_or_scope_fields(self) -> "KeyUpdateFields":
        forbidden = [
            name
            for name in ("key", "key_alias", "team_id")
            if getattr(self, name, None) is not None
        ]
        if forbidden:
            raise ValueError(
                f"Fields not allowed in update_fields for bulk team key updates: "
                f"{forbidden}. `key`/`key_alias` are per-key identifiers; `team_id` "
                f"is the scope guard set at the request top level."
            )
        return self


class BulkUpdateTeamKeysRequest(BaseModel):
    """
    Request for applying one update payload to many keys inside a single team.

    Exactly one of `key_ids` (specific keys in the team) or `all_keys_in_team`
    (every key in the team) must be provided.
    """

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
