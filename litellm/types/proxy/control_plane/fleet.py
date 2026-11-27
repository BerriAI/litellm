from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from litellm.types.proxy.control_plane.dataplanes import UnreachableDataplane
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail
from litellm.types.utils import LiteLLMBaseModel


class FleetModel(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    dataplane_id: str
    model_name: str
    litellm_model: str | None
    provider: str | None
    model_id: str | None
    mode: str | None


class FleetModelGroup(LiteLLMBaseModel):
    """One public model name and every dataplane that serves it."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    dataplanes: tuple[str, ...]


class FleetModelsResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[FleetModel, ...]
    groups: tuple[FleetModelGroup, ...]
    partial: bool
    unreachable: tuple[UnreachableDataplane, ...]


class FleetTeam(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    dataplane_id: str
    team_id: str
    team_alias: str | None
    organization_id: str | None
    spend: float
    max_budget: float | None
    members: int
    models: tuple[str, ...]
    blocked: bool


class FleetTeamsResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[FleetTeam, ...]
    partial: bool
    unreachable: tuple[UnreachableDataplane, ...]


class FleetTeamBudget(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_budget: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    budget_duration: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_budget_field(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("Provide at least one budget field.")
        return self


class FleetTeamAdoption(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    team_alias: str = Field(min_length=1)
    dataplanes: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    fleet_team_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_distinct_dataplanes(self) -> Self:
        if len(set(self.dataplanes)) != len(self.dataplanes):
            raise ValueError("Dataplane IDs must be distinct.")
        return self


class FleetTeamAdoptionConflict(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    dataplane_id: str
    reason: str


class FleetTeamAdoptionProblem(ProblemDetail):
    model_config = ConfigDict(frozen=True)

    conflicts: tuple[FleetTeamAdoptionConflict, ...]


class ManagedFleetTeam(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    fleet_team_id: str
    team_alias: str
    revision: str
    desired: FleetTeamBudget | None


class FleetTeamDataplaneStatus(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    dataplane_id: str
    team_id: str
    desired_revision: str | None
    applied_revision: str | None
    last_error: str | None
    state: Literal["pending", "applied", "drifted", "unreachable"]


class FleetTeamStatus(ManagedFleetTeam):
    dataplanes: tuple[FleetTeamDataplaneStatus, ...]
