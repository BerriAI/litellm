"""The rows the public Model Hub lists: published model groups and opted-in pass-through endpoints."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.proxy._types import PassThroughGenericEndpoint
from litellm.proxy.common_utils.discoverable_model_filter import is_undiscoverable_deployment
from litellm.types.proxy.management_endpoints.model_management_endpoints import ModelGroupInfoProxy

if TYPE_CHECKING:
    from litellm.router import Router

PASS_THROUGH_MODE: Final = "passthrough"
EVERY_MODEL_GROUP: Final = "*"
_SERVING_ENTRY: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class ModelGroupRow:
    info: ModelGroupInfoProxy


@dataclass(frozen=True, slots=True)
class PassThroughRow:
    info: ModelGroupInfoProxy


HubRow: TypeAlias = ModelGroupRow | PassThroughRow


@dataclass(frozen=True, slots=True)
class NoModelsConfigured:
    """Neither a router nor a published pass-through, so the hub has nothing to list."""


def pass_through_row(endpoint: PassThroughGenericEndpoint) -> PassThroughRow:
    return PassThroughRow(
        info=ModelGroupInfoProxy(
            model_group=endpoint.display_name or endpoint.path,
            providers=[],
            mode=PASS_THROUGH_MODE,
            is_public_model_group=True,
            pass_through_path=endpoint.path,
            pass_through_methods=tuple(endpoint.methods) if endpoint.methods is not None else None,
        )
    )


def _serving_pass_through_endpoints() -> Iterator[PassThroughGenericEndpoint]:
    """The endpoints this worker serves, from the in-memory general settings the config sync keeps current."""
    from litellm.proxy.proxy_server import proxy_config

    configured: Final = proxy_config.settings.get("pass_through_endpoints")
    for entry in configured if isinstance(configured, list) else ():
        try:
            yield PassThroughGenericEndpoint.model_validate(_SERVING_ENTRY.validate_python(entry))
        except ValidationError:
            continue


def _publicly_served(deployment: Mapping[str, object]) -> bool:
    match deployment.get("model_info"):
        case {"team_id": str()}:
            return False
        case _:
            return not is_undiscoverable_deployment(deployment)


def _published_by_every_model_group(model_group: str, router: "Router", team_id: str | None) -> bool:
    deployments: Final = router.registry().deployments(model_group, team_id=team_id)
    return not deployments or any(_publicly_served(deployment) for deployment in deployments)


def is_public_model_group(model_group: str, router: "Router", team_id: str | None) -> bool:
    public_names: Final = litellm.public_model_groups or ()
    return model_group in public_names or (
        EVERY_MODEL_GROUP in public_names and _published_by_every_model_group(model_group, router, team_id)
    )


async def _published_model_group_names(public_names: Sequence[str], router: "Router") -> Sequence[str]:
    if EVERY_MODEL_GROUP not in public_names:
        return public_names
    from litellm.proxy.proxy_server import every_model_group_name

    every_name: Final = await every_model_group_name(router)
    return tuple(
        dict.fromkeys(
            (
                *(name for name in every_name if _published_by_every_model_group(name, router, None)),
                *(name for name in public_names if name != EVERY_MODEL_GROUP),
            )
        )
    )


async def _published_model_group_rows() -> tuple[ModelGroupRow, ...] | None:
    """None when there is no router, which differs from a router that publishes nothing."""
    from litellm.proxy.proxy_server import (
        get_model_group_info,
        llm_router,
    )

    if llm_router is None:
        return None
    if litellm.public_model_groups is None:
        return ()
    return tuple(
        ModelGroupRow(info=info)
        for info in get_model_group_info(
            llm_router=llm_router,
            all_models_str=await _published_model_group_names(litellm.public_model_groups, llm_router),
            model_group=None,
        )
    )


async def published_hub_rows() -> tuple[HubRow, ...] | NoModelsConfigured:
    pass_through_rows: Final = tuple(
        pass_through_row(endpoint) for endpoint in _serving_pass_through_endpoints() if endpoint.show_in_model_hub
    )
    model_group_rows: Final = await _published_model_group_rows()
    if model_group_rows is None and not pass_through_rows:
        return NoModelsConfigured()
    return (*(model_group_rows or ()), *pass_through_rows)
