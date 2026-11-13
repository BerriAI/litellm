from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class HookFailure(FrozenModel):
    status: int
    reason: str


class MappedUser(FrozenModel):
    user_id: str
    teams: tuple[str, ...] = ()


class ObservedAgentEvent(FrozenModel):
    request_id: str
    tenant_id: str
    external_agent_id: str
    actor_id: str | None
    frame_type: Literal["prompt", "tool_call"]
    application: str
    model: str | None = None
    tool_names: tuple[str, ...] = ()


class Verdict(FrozenModel):
    action: Literal["allow", "deny"]
    deny_reason: str | None = Field(default=None, max_length=500)
    reference_id: str | None = Field(default=None, max_length=50, pattern=r"^[A-Za-z0-9._:/-]+$")


class SourceDetails(FrozenModel):
    source_id: str
    vendor: Literal["grok_bot"] = "grok_bot"
    display_name: str
    tenant_id: str | None = None
    last_event_at: datetime | None = None
    mode: Literal["shadow", "enforce"] = "shadow"
    default_policy_id: str | None = None


class Source(SourceDetails):
    signing_secret: str | None = Field(default=None, repr=False, exclude=True)
    ingest_key: str | None = Field(default=None, repr=False, exclude=True)
    admin_api_key: str | None = Field(default=None, repr=False, exclude=True)
    previous_signing_secret: str | None = Field(default=None, repr=False, exclude=True)


class Agent(FrozenModel):
    id: str
    source_id: str
    external_agent_id: str
    display_name: str | None = None
    team_id: str | None = None
    owner_user_id: str | None = None
    owner_display_name: str | None = None
    owner_inferred: bool = False
    status: Literal["active", "blocked"] = "active"
    first_seen_at: datetime
    last_seen_at: datetime
    request_count: int
    last_actor_id: str | None = None
    last_model: str | None = None
    issues: tuple[Literal["no team", "no owner", "owner inferred", "unmapped member", "denied recently"], ...] = ()


class Policy(FrozenModel):
    policy_id: str
    source_id: str
    team_id: str | None = None
    blocked_tools: tuple[str, ...] = ()
    allow_unmapped_actors: bool = True
    require_owner: bool = False


class Decision(FrozenModel):
    request_id: str
    source_id: str
    external_agent_id: str
    actor_id: str | None = None
    user_id: str | None = None
    user_display_name: str | None = None
    agent_display_name: str | None = None
    team_id: str | None = None
    frame_type: str
    tool_names: tuple[str, ...] = ()
    verdict: Literal["allow", "deny"]
    would_verdict: Literal["allow", "deny"]
    reason: str | None = None
    enforced: bool
    policy_id: str | None = None
    created_at: datetime


class SourceCreate(FrozenModel):
    display_name: str = Field(default="GrokBot", min_length=1, max_length=200)
    signing_secret: SecretStr | None = None
    admin_api_key: SecretStr | None = None
    mode: Literal["shadow", "enforce"] = "shadow"


class SourcePatch(FrozenModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    signing_secret: SecretStr | None = None
    admin_api_key: SecretStr | None = None
    mode: Literal["shadow", "enforce"] | None = None


class MemberSyncResult(FrozenModel):
    matched: int
    unmatched: int
    already_mapped: int


class MemberSyncRequest(FrozenModel):
    pass


class SourceSaved(SourceDetails):
    sync_result: MemberSyncResult | None = None


class SourceConnection(SourceSaved):
    ingest_key: str = Field(repr=False)
    hook_path: str


class AgentPatch(FrozenModel):
    agent_ids: tuple[str, ...] = Field(min_length=1, max_length=1000)
    team_id: str | None = None
    owner_user_id: str | None = None
    status: Literal["active", "blocked"] | None = None
    display_name: str | None = Field(default=None, max_length=200)


class PolicyPut(FrozenModel):
    source_id: str
    team_id: str | None = None
    blocked_tools: tuple[str, ...] = Field(default=(), max_length=1000)
    allow_unmapped_actors: bool = True
    require_owner: bool = False


class ActorMapping(FrozenModel):
    external_actor_id: str = Field(min_length=1)
    user_id: str


class ActorsPut(FrozenModel):
    source_id: str
    mappings: tuple[ActorMapping, ...] = Field(min_length=1, max_length=1000)


class ExternalAgentSummary(FrozenModel):
    discovered: int
    covered_by_team_policy: int
    needs_owner: int
    unmapped_actors: int
    denied_24h: int
    last_event_at: datetime | None
    needs_attention: int = 0


class Updated(FrozenModel):
    updated: int
