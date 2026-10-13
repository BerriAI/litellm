from pydantic import ConfigDict

from litellm.types.proxy.control_plane.dataplanes import UnreachableDataplane
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
