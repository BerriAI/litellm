"""Client for budget e2e tests: the shared Gateway plus budget-bearing entity
management (user / team / team-member / org / customer / tag / budget-table) and
info reads.

Over-budget surfaces as a ``budget_exceeded`` error; ``is_budget_block`` detects it
on a chat outcome. Create methods return the new id and raise on failure; tests
register the matching delete with ``resources.defer(...)`` for cleanup. The request
and response models are co-located here because only this suite uses them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import AliasPath, BaseModel, Field, RootModel

from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, Result, StreamingResponse, Success, unwrap
from models import (
    AnthropicMessagesBody,
    BudgetWindow,
    ChatBody,
    ChatMessage,
    ChatMetadata,
    KeyGenerateBody,
    ModelBudgetEntry,
)

_TEAM_READY_ATTEMPTS = 15
_TEAM_READY_SLEEP_SECONDS = 0.4


class UserNewBody(BaseModel):
    max_budget: float


class UserNewResponse(BaseModel):
    user_id: str


class UserInfoParams(BaseModel):
    user_id: str


class UserInfoRow(BaseModel):
    spend: float | None = None
    max_budget: float | None = None


class UserInfoResponse(BaseModel):
    user_info: UserInfoRow | None = None


class UserDeleteBody(BaseModel):
    user_ids: list[str]


class CustomerNewBody(BaseModel):
    user_id: str
    max_budget: float


class OrgNewBody(BaseModel):
    organization_alias: str
    max_budget: float


class OrgNewResponse(BaseModel):
    organization_id: str


class OrgDeleteBody(BaseModel):
    organization_ids: list[str]


class TeamMember(BaseModel):
    role: str
    user_id: str


class TeamNewBody(BaseModel):
    team_alias: str
    max_budget: float | None = None
    organization_id: str | None = None
    budget_limits: list[BudgetWindow] | None = None


class TeamNewResponse(BaseModel):
    team_id: str


class TeamDeleteBody(BaseModel):
    team_ids: list[str]


class TeamMemberAddBody(BaseModel):
    team_id: str
    member: TeamMember
    max_budget_in_team: float | None = None


class TeamMemberUpdateBody(BaseModel):
    team_id: str
    user_id: str
    max_budget_in_team: float | None = None
    budget_duration: str | None = None


class TeamMembershipRow(BaseModel):
    user_id: str | None = None
    budget_reset_at: str | None = Field(
        default=None,
        validation_alias=AliasPath("litellm_budget_table", "budget_reset_at"),
    )


class TeamInfoParams(BaseModel):
    team_id: str


class TeamInfoResponse(BaseModel):
    team_memberships: list[TeamMembershipRow] = []


class TagNewBody(BaseModel):
    name: str
    max_budget: float


class TagDeleteBody(BaseModel):
    name: str


class BudgetNewBody(BaseModel):
    max_budget: float
    soft_budget: float | None = None
    budget_duration: str | None = None


class BudgetNewResponse(BaseModel):
    budget_id: str


class BudgetDeleteBody(BaseModel):
    id: str


class BudgetInfoBody(BaseModel):
    budgets: list[str]


class BudgetRow(BaseModel):
    budget_id: str | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    budget_reset_at: str | None = None


class BudgetInfoResponse(RootModel[list[BudgetRow]]):
    pass


def is_budget_block(result: StreamingResponse) -> bool:
    """True if the call was rejected for being over budget (vs a provider error)."""
    return not result.ok and "budget_exceeded" in result.body


def model_budget(model: str, limit: float, period: str = "30d") -> dict[str, ModelBudgetEntry]:
    """A model_max_budget entry: per-model cap with a reset window."""
    return {model: ModelBudgetEntry(budget_limit=limit, time_period=period)}


@dataclass(frozen=True, slots=True)
class BudgetClient:
    gateway: Gateway

    # ---- generic key ops (delegate to the shared Gateway) ---------------

    def generate_key(
        self,
        *,
        models: list[str] | None = None,
        max_budget: float | None = None,
        soft_budget: float | None = None,
        budget_duration: str | None = None,
        budget_id: str | None = None,
        user_id: str | None = None,
        team_id: str | None = None,
        model_max_budget: dict[str, ModelBudgetEntry] | None = None,
        budget_fallbacks: dict[str, list[str]] | None = None,
        budget_limits: list[BudgetWindow] | None = None,
    ) -> str:
        return self.gateway.generate_key(
            KeyGenerateBody(
                models=models or [],
                max_budget=max_budget,
                soft_budget=soft_budget,
                budget_duration=budget_duration,
                budget_id=budget_id,
                user_id=user_id,
                team_id=team_id,
                model_max_budget=model_max_budget,
                budget_fallbacks=budget_fallbacks,
                budget_limits=budget_limits,
            )
        )

    def delete_key(self, key: str) -> None:
        self.gateway.delete_key(key)

    def delete_customers(self, user_ids: list[str]) -> None:
        self.gateway.delete_customers(user_ids)

    # ---- chat (raw HTTP outcome: a budget block surfaces as a non-2xx) --

    def chat(
        self,
        key: str,
        model: str,
        content: str,
        *,
        max_tokens: int | None = None,
        user: str | None = None,
        tags: list[str] | None = None,
    ) -> StreamingResponse:
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
                user=user,
                metadata=ChatMetadata(tags=tags) if tags else None,
            ),
        )

    def messages(
        self,
        key: str,
        model: str,
        content: str,
        *,
        max_tokens: int = 16,
    ) -> StreamingResponse:
        return self.gateway.transport.send(
            "/v1/messages",
            headers=self.gateway.transport.bearer(key),
            json=AnthropicMessagesBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
            ),
        )

    # ---- internal user --------------------------------------------------

    def create_user(self, *, max_budget: float) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/user/new",
                headers=self.gateway.transport.master,
                json=UserNewBody(max_budget=max_budget),
                response_type=UserNewResponse,
            )
        ).user_id

    def delete_user(self, user_id: str) -> None:
        _ = self.gateway.transport.post(
            "/user/delete",
            headers=self.gateway.transport.master,
            json=UserDeleteBody(user_ids=[user_id]),
            response_type=NoBody,
        )

    def user_info(self, user_id: str) -> UserInfoRow | None:
        result = self.gateway.transport.get(
            "/user/info",
            headers=self.gateway.transport.master,
            params=UserInfoParams(user_id=user_id),
            response_type=UserInfoResponse,
        )
        match result:
            case Success(data=data):
                return data.user_info
            case _:
                return None

    # ---- customer / end-user -------------------------------------------

    def create_customer(self, customer_id: str, *, max_budget: float) -> str:
        resp = self.gateway.transport.send(
            "/customer/new",
            headers=self.gateway.transport.master,
            json=CustomerNewBody(user_id=customer_id, max_budget=max_budget),
        )
        assert resp.ok, resp.body
        return customer_id

    # ---- organization ---------------------------------------------------

    def create_org(self, *, max_budget: float, alias: str) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/organization/new",
                headers=self.gateway.transport.master,
                json=OrgNewBody(organization_alias=alias, max_budget=max_budget),
                response_type=OrgNewResponse,
            )
        ).organization_id

    def delete_org(self, org_id: str) -> None:
        _ = self.gateway.transport.delete(
            "/organization/delete",
            headers=self.gateway.transport.master,
            json=OrgDeleteBody(organization_ids=[org_id]),
            response_type=NoBody,
        )

    # ---- team -----------------------------------------------------------

    def create_team(
        self,
        *,
        alias: str,
        max_budget: float | None = None,
        organization_id: str | None = None,
        budget_limits: list[BudgetWindow] | None = None,
    ) -> str:
        team_id = unwrap(
            self.gateway.transport.post(
                "/team/new",
                headers=self.gateway.transport.master,
                json=TeamNewBody(
                    team_alias=alias,
                    max_budget=max_budget,
                    organization_id=organization_id,
                    budget_limits=budget_limits,
                ),
                response_type=TeamNewResponse,
            )
        ).team_id
        self._wait_for_team(team_id)
        return team_id

    def delete_team(self, team_id: str) -> None:
        _ = self.gateway.transport.post(
            "/team/delete",
            headers=self.gateway.transport.master,
            json=TeamDeleteBody(team_ids=[team_id]),
            response_type=NoBody,
        )

    def _wait_for_team(self, team_id: str) -> None:
        last: Result[TeamInfoResponse] | None = None
        for _ in range(_TEAM_READY_ATTEMPTS):
            last = self.gateway.transport.get(
                "/team/info",
                headers=self.gateway.transport.master,
                params=TeamInfoParams(team_id=team_id),
                response_type=TeamInfoResponse,
            )
            match last:
                case Success():
                    return
                case _:
                    time.sleep(_TEAM_READY_SLEEP_SECONDS)
        assert last is not None
        raise AssertionError(last)

    def add_team_member(self, team_id: str, user_id: str, *, max_budget_in_team: float | None = None) -> None:
        last_body = ""
        for attempt in range(_TEAM_READY_ATTEMPTS):
            resp = self.gateway.transport.send(
                "/team/member_add",
                headers=self.gateway.transport.master,
                json=TeamMemberAddBody(
                    team_id=team_id,
                    member=TeamMember(role="user", user_id=user_id),
                    max_budget_in_team=max_budget_in_team,
                ),
            )
            if resp.ok:
                return
            last_body = resp.body
            if "doesn't exist" in resp.body and attempt + 1 < _TEAM_READY_ATTEMPTS:
                time.sleep(_TEAM_READY_SLEEP_SECONDS)
                continue
            break
        assert False, last_body

    def update_team_member(
        self,
        team_id: str,
        user_id: str,
        *,
        max_budget_in_team: float | None = None,
        budget_duration: str | None = None,
    ) -> None:
        resp = self.gateway.transport.send(
            "/team/member_update",
            headers=self.gateway.transport.master,
            json=TeamMemberUpdateBody(
                team_id=team_id,
                user_id=user_id,
                max_budget_in_team=max_budget_in_team,
                budget_duration=budget_duration,
            ),
        )
        assert resp.ok, resp.body

    def member_budget_reset_at(self, team_id: str, user_id: str) -> str | None:
        """The member's per-team budget_reset_at as /team/info reports it, or None if
        no reset is scheduled. The reset job advances this each time the window
        elapses; a job that skips the row leaves it pinned forever."""
        result = self.gateway.transport.get(
            "/team/info",
            headers=self.gateway.transport.master,
            params=TeamInfoParams(team_id=team_id),
            response_type=TeamInfoResponse,
        )
        match result:
            case Success(data=data):
                return next(
                    (row.budget_reset_at for row in data.team_memberships if row.user_id == user_id),
                    None,
                )
            case _:
                return None

    # ---- tag ------------------------------------------------------------

    def create_tag(self, name: str, *, max_budget: float) -> str:
        resp = self.gateway.transport.send(
            "/tag/new",
            headers=self.gateway.transport.master,
            json=TagNewBody(name=name, max_budget=max_budget),
        )
        assert resp.ok, resp.body
        return name

    def delete_tag(self, name: str) -> None:
        _ = self.gateway.transport.post(
            "/tag/delete",
            headers=self.gateway.transport.master,
            json=TagDeleteBody(name=name),
            response_type=NoBody,
        )

    # ---- budget table ---------------------------------------------------

    def create_budget(
        self,
        *,
        max_budget: float,
        soft_budget: float | None = None,
        budget_duration: str | None = None,
    ) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/budget/new",
                headers=self.gateway.transport.master,
                json=BudgetNewBody(
                    max_budget=max_budget,
                    soft_budget=soft_budget,
                    budget_duration=budget_duration,
                ),
                response_type=BudgetNewResponse,
            )
        ).budget_id

    def delete_budget(self, budget_id: str) -> None:
        _ = self.gateway.transport.post(
            "/budget/delete",
            headers=self.gateway.transport.master,
            json=BudgetDeleteBody(id=budget_id),
            response_type=NoBody,
        )

    def budget_info(self, budget_id: str) -> tuple[BudgetRow, ...]:
        result = self.gateway.transport.post(
            "/budget/info",
            headers=self.gateway.transport.master,
            json=BudgetInfoBody(budgets=[budget_id]),
            response_type=BudgetInfoResponse,
        )
        match result:
            case Success(data=data):
                return tuple(data.root)
            case _:
                return ()


def build_client() -> BudgetClient:
    return BudgetClient(gateway=build_gateway())
