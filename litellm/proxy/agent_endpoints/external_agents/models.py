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


class Source(FrozenModel):
    source_id: str
    vendor: Literal["grok_bot"] = "grok_bot"
    display_name: str
    tenant_id: str | None = None
    signing_secret: str = Field(repr=False, exclude=True)
    previous_signing_secret: str | None = Field(default=None, repr=False, exclude=True)
    mode: Literal["shadow", "enforce"] = "shadow"
    default_policy_id: str | None = None


class Agent(FrozenModel):
    id: str
    source_id: str
    external_agent_id: str
    display_name: str | None = None
    team_id: str | None = None
    owner_user_id: str | None = None
    owner_inferred: bool = False
    status: Literal["active", "blocked"] = "active"
    first_seen_at: datetime
    last_seen_at: datetime
    request_count: int
    last_actor_id: str | None = None
    last_model: str | None = None


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
    display_name: str = Field(min_length=1, max_length=200)
    signing_secret: SecretStr
    mode: Literal["shadow", "enforce"] = "shadow"


class SourcePatch(FrozenModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    signing_secret: SecretStr | None = None
    mode: Literal["shadow", "enforce"] | None = None


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


class Summary(FrozenModel):
    discovered: int
    covered_by_team_policy: int
    needs_owner: int
    unmapped_actors: int
    denied_24h: int
    last_event_at: datetime | None


class Updated(FrozenModel):
    updated: int
