from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from fastapi import status
from pydantic import ConfigDict, ValidationError

import litellm
from litellm.proxy._types import (
    ConfigGeneralSettings,
    ModelAccessDeniedProxyException,
    ProxyErrorTypes,
)
from litellm.proxy.auth.model_access_denied import model_access_denied_client_message
from litellm.types.llms.base_model import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.router import Router

_SETTING: Final = "model_access_group_deny_by_default"


class _DeploymentModelInfo(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    access_groups: tuple[str, ...] | None = None


class _DeploymentRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    model_name: str
    model_info: _DeploymentModelInfo | None = None

    @property
    def groups(self) -> frozenset[str]:
        return frozenset(self.model_info.access_groups or ()) if self.model_info is not None else frozenset()


def model_access_group_deny_by_default(general_settings: Mapping[str, object]) -> bool:
    """
    Startup rejects a non-boolean value from the config file. A non-boolean value that reaches
    general_settings another way enables the policy, so it fails closed
    """
    value: Final = general_settings.get(_SETTING, False)
    if isinstance(value, bool):
        return value
    try:
        return ConfigGeneralSettings.model_validate(
            MappingProxyType({_SETTING: value})
        ).model_access_group_deny_by_default
    except ValidationError:
        return True


def _alias_chain(
    model: str, team_model_aliases: Mapping[str, str] | None, key_model_aliases: Mapping[str, str] | None
) -> Iterator[str]:
    after_team: Final = team_model_aliases.get(model, model) if team_model_aliases else model
    after_key: Final = key_model_aliases.get(after_team, after_team) if key_model_aliases else after_team
    yield from (model, after_team, after_key, litellm.model_alias_map.get(after_key, after_key))


def _alias_chains(
    model: str | Sequence[str],
    team_model_aliases: Mapping[str, str] | None,
    key_model_aliases: Mapping[str, str] | None,
) -> Iterator[str]:
    for requested in (model,) if isinstance(model, str) else model:
        yield from _alias_chain(requested, team_model_aliases, key_model_aliases)


def requested_model_names(
    model: str | Sequence[str],
    team_model_aliases: Mapping[str, str] | None = None,
    key_model_aliases: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """The requested models plus every model their team, key and proxy aliases point at"""
    return tuple(dict.fromkeys(_alias_chains(model, team_model_aliases, key_model_aliases)))


def _rows(rows: Iterable[object]) -> Iterator[_DeploymentRow]:
    for row in rows:
        yield _DeploymentRow.model_validate(row)


def _routing_group_member_rows(name: str, llm_router: "Router", team_id: str | None) -> Iterator[_DeploymentRow]:
    group: Final = llm_router.get_routing_group(llm_router.get_model_from_alias(model=name) or name)
    for member in group.models if group is not None else ():
        yield from _rows(llm_router.get_model_list(model_name=member, team_id=team_id) or ())


def _reachable_deployments(name: str, llm_router: "Router", team_id: str | None) -> Iterator[_DeploymentRow]:
    """Every deployment a request for `name` can be served by: its model group (aliases and wildcards
    included), a routing group's members, a deployment id, or a deployment's litellm_params.model"""
    yield from _rows(llm_router.get_model_list(model_name=name, team_id=team_id) or ())
    yield from _routing_group_member_rows(name, llm_router, team_id)
    deployment: Final = llm_router.get_deployment(model_id=name) if llm_router.has_model_id(name) else None
    if deployment is not None:
        yield _DeploymentRow.model_validate(deployment.model_dump(exclude_none=True))
    if name in llm_router.deployment_names:
        yield from _rows(row for row in llm_router.get_model_list() or () if row["litellm_params"].get("model") == name)


def ungranted_access_group_models(
    names: Sequence[str],
    granted_models: Sequence[str],
    llm_router: "Router | None",
    team_id: str | None,
) -> tuple[str, ...]:
    """
    The names that can reach a deployment in a model access group which `granted_models` does not name,
    by group name, the deployment's model name, or the requested name. Names match exactly, so a blanket
    grant (empty list, "*", all-proxy-models, all-team-models) or a wildcard pattern names nothing
    """
    if llm_router is None:
        return ()
    explicit: Final = frozenset(granted_models)
    return tuple(name for name in names if not _is_explicitly_granted(name, explicit, llm_router, team_id))


def _is_explicitly_granted(name: str, explicit: frozenset[str], llm_router: "Router", team_id: str | None) -> bool:
    return name in explicit or all(
        not row.groups or row.model_name in explicit or not explicit.isdisjoint(row.groups)
        for row in _reachable_deployments(name, llm_router, team_id)
    )


def access_group_model_denied(
    model: str | Sequence[str], ungranted: Sequence[str], object_type: Literal["key", "user", "team"]
) -> ModelAccessDeniedProxyException:
    return ModelAccessDeniedProxyException(
        message=model_access_denied_client_message(model=model if isinstance(model, str) else list(model)),
        internal_message=(
            f"{object_type} not allowed to access model. {', '.join(ungranted)} reaches a model access group and "
            f"general_settings.{_SETTING} requires the {object_type} to list that group or model explicitly"
        ),
        type=ProxyErrorTypes.get_model_access_error_type_for_object(object_type=object_type),
        param="model",
        code=status.HTTP_403_FORBIDDEN,
    )
