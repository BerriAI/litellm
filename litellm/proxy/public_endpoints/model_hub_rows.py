"""The rows the public Model Hub lists: published model groups and opted-in pass-through endpoints."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.proxy._types import PassThroughGenericEndpoint
from litellm.types.proxy.management_endpoints.model_management_endpoints import ModelGroupInfoProxy

PASS_THROUGH_MODE: Final = "passthrough"
_SERVING_ENTRIES: Final = TypeAdapter(list[dict[str, object]])


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
            pass_through_methods=endpoint.methods,
        )
    )


def _serving_pass_through_endpoints() -> Iterator[PassThroughGenericEndpoint]:
    """The endpoints this worker serves, from the in-memory general settings the config sync keeps current."""
    from litellm.proxy.proxy_server import proxy_config

    configured: Final = proxy_config.settings.get("pass_through_endpoints")
    for entry in _SERVING_ENTRIES.validate_python(configured) if isinstance(configured, list) else ():
        try:
            yield PassThroughGenericEndpoint.model_validate(entry)
        except ValidationError:
            continue


def _published_model_group_rows() -> tuple[ModelGroupRow, ...] | None:
    """None when there is no router, which differs from a router that publishes nothing."""
    from litellm.proxy.proxy_server import (
        _get_model_group_info,  # pyright: ignore[reportPrivateUsage]  # the hub routes have always read it here
        llm_router,
    )

    if llm_router is None:
        return None
    if litellm.public_model_groups is None:
        return ()
    return tuple(
        ModelGroupRow(info=info)
        for info in _get_model_group_info(
            llm_router=llm_router, all_models_str=litellm.public_model_groups, model_group=None
        )
    )


def published_hub_rows() -> tuple[HubRow, ...] | NoModelsConfigured:
    pass_through_rows: Final = tuple(
        pass_through_row(endpoint) for endpoint in _serving_pass_through_endpoints() if endpoint.show_in_model_hub
    )
    model_group_rows: Final = _published_model_group_rows()
    if model_group_rows is None and not pass_through_rows:
        return NoModelsConfigured()
    return (*(model_group_rows or ()), *pass_through_rows)
