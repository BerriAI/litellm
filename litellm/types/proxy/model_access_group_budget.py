"""The model access group budget state auth and the spend reservation path share."""

from __future__ import annotations

from pydantic import BaseModel


class ModelAccessGroupBudget(BaseModel):
    """One model access group's budget, flattened out of its joined ``LiteLLM_ModelAccessGroupBudgetTable`` row.

    Both readers want only the recorded spend and the ceiling, and this sits on the per-request hot
    path behind a cache, so the linked budget row is collapsed to ``max_budget`` rather than cached
    whole. ``spend`` is the DB-recorded value, which lags the live counter and is only ever a
    fallback for it.
    """

    access_group_name: str
    spend: float = 0.0
    max_budget: float | None = None
