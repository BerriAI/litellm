from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from litellm.proxy._types import Litellm_EntityType


class BulkUpdateKeyRequestItem(BaseModel):
    """Individual key update request item"""

    key: str  # Key identifier (token)
    budget_id: str | None = None  # Budget ID associated with the key
    max_budget: float | None = None  # Max budget for key
    team_id: str | None = None  # Team ID associated with key
    tags: list[str] | None = None  # Tags for organizing keys


class BulkUpdateKeyRequest(BaseModel):
    """Request for bulk key updates"""

    keys: list[BulkUpdateKeyRequestItem]


class SuccessfulKeyUpdate(BaseModel):
    """Successfully updated key with its updated information"""

    key: str
    key_info: dict[str, Any]


class FailedKeyUpdate(BaseModel):
    """Failed key update with reason"""

    key: str
    key_info: dict[str, Any] | None = None
    failed_reason: str


class BulkUpdateKeyResponse(BaseModel):
    """Response for bulk key update operations"""

    total_requested: int
    successful_updates: list[SuccessfulKeyUpdate]
    failed_updates: list[FailedKeyUpdate]


class KeyUpdateFields(BaseModel):
    """Allowlist of bulk-broadcastable fields for /team/key/bulk_update; `extra="forbid"` blocks RBAC/ownership/scope mutations even by team admins."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Budgets
    max_budget: float | None = None
    budget_id: str | None = None
    budget_duration: str | None = None
    budget_limits: list[Any] | None = None
    model_max_budget: dict[str, Any] | None = None

    # Rate limits
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    model_tpm_limit: dict[str, Any] | None = None
    model_rpm_limit: dict[str, Any] | None = None
    max_parallel_requests: int | None = None
    rpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"] | None = None
    tpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"] | None = None

    # Temporary budget grants (auto-expire). `spend` deliberately omitted — bulk-zeroing it bypasses budget enforcement; admin-only via /key/update.
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None

    # Expiry
    duration: str | None = None

    # Operational metadata
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_temp_budget(self) -> "KeyUpdateFields":
        if self.temp_budget_increase is not None or self.temp_budget_expiry is not None:
            if self.temp_budget_increase is None or self.temp_budget_expiry is None:
                raise ValueError("temp_budget_increase and temp_budget_expiry must be set together")
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
    key_ids: list[str] | None = None
    all_keys_in_team: bool = False
    update_fields: KeyUpdateFields

    @model_validator(mode="after")
    def validate_selection(self) -> "BulkUpdateTeamKeysRequest":
        has_key_ids: Final = self.key_ids is not None and len(self.key_ids) > 0
        if has_key_ids and self.all_keys_in_team:
            raise ValueError("Provide either `key_ids` or `all_keys_in_team=True`, not both.")
        if not has_key_ids and not self.all_keys_in_team:
            raise ValueError("Must provide either `key_ids` (non-empty) or `all_keys_in_team=True`.")
        return self


BudgetScope = Literal[
    "proxy",
    "key",
    "key_window",
    "key_model",
    "team",
    "team_window",
    "team_member",
    "user",
    "organization",
    "project",
    "tag",
    "end_user",
    "end_user_model",
]

BudgetEnforcement = Literal["hard", "soft", "throttled"]

BudgetComparison = Literal[">=", ">"]

BudgetStatus = Literal["unlimited", "ok", "exceeded"]

BudgetNoteCode = Literal[
    "alert_only",
    "custom_auth_may_override_end_user_cap",
    "custom_auth_skips_read_time_checks",
    "end_user_route_only",
    "per_model_counters",
    "project_spend_not_tracked",
    "request_tags_add_budgets",
    "reservation_blocks_at_limit",
    "rolling_window",
    "throttled_instead_of_blocked",
    "user_budget_not_applied_to_team_key",
]

BudgetNoteSeverity = Literal["info", "warning"]

BudgetSpendState = Literal["live", "no_counter", "unavailable"]


class KeyBudgetNote(BaseModel):
    """
    One caveat about a budget row.

    ``code`` is the contract: map it to whatever treatment the caveat deserves. ``text`` is free to be
    reworded and must not be matched on. ``severity`` exists for the code a client has not been taught
    yet, since this union grows, and it turns on whether the row already carries the fact in a field:
    ``info`` means the note only explains something the row states anyway, like ``enforcement``,
    ``comparison`` or ``spend_state``, and ``warning`` means the note alone carries it, so the row
    cannot be taken at face value without reading it.
    """

    model_config = ConfigDict(frozen=True)

    code: BudgetNoteCode
    severity: BudgetNoteSeverity
    text: str


class KeyBudgetEntry(BaseModel):
    """One budget that can gate requests made with a key, with its live spend."""

    scope: BudgetScope
    entity_type: Litellm_EntityType
    entity_id: str | None = None
    entity_label: str | None = None
    enforcement: BudgetEnforcement
    max_budget: float | None = None
    spend: float | None = None
    spend_state: BudgetSpendState
    remaining: float | None = None
    comparison: BudgetComparison
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    window_start: datetime | None = None
    source: str
    status: BudgetStatus
    notes: tuple[KeyBudgetNote, ...] = ()


class KeyBudgetsResponse(BaseModel):
    """Every budget that applies to one key, including the ones left unconfigured."""

    key: str
    budgets: tuple[KeyBudgetEntry, ...]
