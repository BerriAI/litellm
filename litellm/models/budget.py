"""
Budget table model.

Canonical definition for ``litellm_budgettable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from typing import List, Optional

from pydantic import ConfigDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_BudgetTable(LiteLLMPydanticObjectBase):
    """Represents user-controllable params for a LiteLLM_BudgetTable record.

    Budget-write paths use `model_fields.keys()` on this class as an allowlist
    for user input. Keep server-managed fields (e.g. `budget_reset_at`) on
    `LiteLLM_BudgetTableFull` so they aren't user-settable.
    """

    budget_id: Optional[str] = None
    soft_budget: Optional[float] = None
    max_budget: Optional[float] = None
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    model_max_budget: Optional[dict] = None
    budget_duration: Optional[str] = None
    allowed_models: Optional[List[str]] = (
        None  # per-member model scope; empty = inherit team models
    )

    model_config = ConfigDict(protected_namespaces=())
