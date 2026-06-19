"""Client for budget e2e tests: the shared ProxyClient plus budget-bearing entity
management (user / team / team-member / org / customer / tag / budget-table) and
info reads.

Over-budget surfaces as a ``budget_exceeded`` error; ``is_budget_block`` detects it
on a CallResult. Create methods return the new id and raise on failure; tests
register the matching delete with ``resources.defer(...)`` for cleanup.
"""

from typing import Dict, Optional

import requests
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from proxy_client import CallResult, ProxyClient, auth_headers, proxy_client_kwargs


def is_budget_block(result: CallResult) -> bool:
    """True if the call was rejected for being over budget (vs a provider error)."""
    return not result.ok and "budget_exceeded" in result.body


def model_budget(model: str, limit: float, period: str = "30d") -> dict:
    """A model_max_budget dict entry: per-model cap with a reset window."""
    return {model: {"budget_limit": limit, "time_period": period}}


@dataclass(frozen=True, slots=True)
class BudgetRow:
    """A /budget/info row: only the fields tests assert on, pydantic ignores the rest."""

    budget_id: Optional[str] = None
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[str] = None


_BUDGET_ROWS = TypeAdapter(tuple[BudgetRow, ...])


class BudgetClient(ProxyClient):
    def _post(self, path: str, body: Dict[str, object]) -> Dict[str, object]:
        resp = requests.post(
            f"{self._base_url}{path}",
            headers=auth_headers(self._master_key),
            json=body,
            timeout=self._request_timeout,
        )
        resp.raise_for_status()
        data = resp.json() if resp.text else {}
        # delete routes return a bare count, not an object; callers ignore it.
        return data if isinstance(data, dict) else {}

    def _delete(self, path: str, body: Dict[str, object]) -> None:
        resp = requests.delete(
            f"{self._base_url}{path}",
            headers=auth_headers(self._master_key),
            json=body,
            timeout=self._request_timeout,
        )
        resp.raise_for_status()

    # ---- internal user --------------------------------------------------

    def create_user(self, *, max_budget: float, budget_duration: Optional[str] = None) -> str:
        body: Dict[str, object] = {"max_budget": max_budget}
        if budget_duration is not None:
            body["budget_duration"] = budget_duration
        return str(self._post("/user/new", body)["user_id"])

    def delete_user(self, user_id: str) -> None:
        self._post("/user/delete", {"user_ids": [user_id]})

    # ---- customer / end-user -------------------------------------------

    def create_customer(self, customer_id: str, *, max_budget: float) -> str:
        self._post("/customer/new", {"user_id": customer_id, "max_budget": max_budget})
        return customer_id

    # ---- organization ---------------------------------------------------

    def create_org(self, *, max_budget: float, alias: str) -> str:
        return str(
            self._post(
                "/organization/new",
                {"organization_alias": alias, "max_budget": max_budget},
            )["organization_id"]
        )

    def delete_org(self, org_id: str) -> None:
        self._delete("/organization/delete", {"organization_ids": [org_id]})

    # ---- team -----------------------------------------------------------

    def create_team(
        self,
        *,
        alias: str,
        max_budget: Optional[float] = None,
        organization_id: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> str:
        body: Dict[str, object] = {"team_alias": alias}
        if max_budget is not None:
            body["max_budget"] = max_budget
        if organization_id is not None:
            body["organization_id"] = organization_id
        if extra:
            body.update(extra)
        return str(self._post("/team/new", body)["team_id"])

    def delete_team(self, team_id: str) -> None:
        self._post("/team/delete", {"team_ids": [team_id]})

    def add_team_member(self, team_id: str, user_id: str, *, max_budget_in_team: Optional[float] = None) -> None:
        body: Dict[str, object] = {
            "team_id": team_id,
            "member": {"role": "user", "user_id": user_id},
        }
        if max_budget_in_team is not None:
            body["max_budget_in_team"] = max_budget_in_team
        self._post("/team/member_add", body)

    # ---- tag ------------------------------------------------------------

    def create_tag(self, name: str, *, max_budget: float) -> str:
        self._post("/tag/new", {"name": name, "max_budget": max_budget})
        return name

    def delete_tag(self, name: str) -> None:
        self._post("/tag/delete", {"name": name})

    # ---- budget table ---------------------------------------------------

    def create_budget(
        self,
        *,
        max_budget: float,
        soft_budget: Optional[float] = None,
        budget_duration: Optional[str] = None,
    ) -> str:
        body: Dict[str, object] = {"max_budget": max_budget}
        if soft_budget is not None:
            body["soft_budget"] = soft_budget
        if budget_duration is not None:
            body["budget_duration"] = budget_duration
        return str(self._post("/budget/new", body)["budget_id"])

    def delete_budget(self, budget_id: str) -> None:
        self._post("/budget/delete", {"id": budget_id})

    def budget_info(self, budget_id: str) -> tuple[BudgetRow, ...]:
        resp = requests.post(
            f"{self._base_url}/budget/info",
            headers=auth_headers(self._master_key),
            json={"budgets": [budget_id]},
            timeout=self._request_timeout,
        )
        resp.raise_for_status()
        return _BUDGET_ROWS.validate_python(resp.json())


def build_client() -> BudgetClient:
    return BudgetClient(**proxy_client_kwargs())
