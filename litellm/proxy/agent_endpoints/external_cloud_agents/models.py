from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)


class Failure(FrozenModel):
    status: int
    message: str


class Allow(FrozenModel):
    action: Literal["allow"] = "allow"


class SourceRecord(FrozenModel):
    source_id: str = Field(alias="id")
    ingest_key: str = Field(repr=False, exclude=True)
    tenant_id: str | None = None
    admin_api_key: str | None = Field(default=None, repr=False, exclude=True)
    last_spend_sync_at: datetime | None = None


class Source(FrozenModel):
    source_id: str
    vendor: str
    display_name: str
    tenant_id: str | None
    bot_count: int
    connected: bool
    has_admin_api_key: bool
    last_event_at: datetime | None
    last_spend_sync_at: datetime | None
    last_spend_sync_error: str | None
    created_at: datetime
    created_by: str


class Connection(FrozenModel):
    source_id: str
    hook_path: str
    ingest_key: str = Field(repr=False)


class AdminKey(FrozenModel):
    api_key: SecretStr


class Updated(FrozenModel):
    status: Literal["ok"] = "ok"


class Agent(FrozenModel):
    source_id: str
    external_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    request_count: int
    last_actor_id: str | None
    last_model: str | None
    tools: tuple[str, ...]


class AgentPage(FrozenModel):
    data: tuple[Agent, ...]
    page: int
    page_size: int
    total: int


class Count(FrozenModel):
    total: int


class DailySpend(FrozenModel):
    vendor_user_id: str
    email: str | None = None
    display_name: str | None = None
    day: date
    cost_micros: int
    prompt_tokens: int
    completion_tokens: int
    requests: int


class SpendTotals(FrozenModel):
    grok_cost_usd: Decimal = Decimal(0)
    litellm_gateway_spend_usd: Decimal = Decimal(0)
    tokens: int = 0
    requests: int = 0


class MemberSpend(SpendTotals):
    vendor_user_id: str
    email: str | None = None
    display_name: str | None = None
    litellm_user_id: str | None = None


class MatchedSpend(MemberSpend):
    team_ids: tuple[str, ...] = ()
    team_names: tuple[str, ...] = ()


class SpendTeam(FrozenModel):
    team_id: str
    team_name: str


class TeamSpend(SpendTotals):
    team_id: str | None
    team_name: str
    teams: tuple[SpendTeam, ...] = ()


class SpendReport(FrozenModel):
    source_id: str
    days: int
    start_day: date
    end_day: date
    members: tuple[MemberSpend, ...]
    teams: tuple[TeamSpend, ...] = Field(
        description="Members with exactly one team are attributed to that team. Multi-team members are grouped "
        "under Multiple teams with their memberships listed, without allocating spend across teams. "
        "Unmatched, ambiguous-email and teamless members are Unassigned. Gateway spend is counted once per "
        "LiteLLM user, on the lexicographically first vendor identity when several identities match that user."
    )
    totals: SpendTotals


class SyncResult(FrozenModel):
    status: Literal["ok", "already_running"] = "ok"
    rows: int = 0
