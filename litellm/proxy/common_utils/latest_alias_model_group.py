from collections.abc import Iterable, Iterator, Mapping
from itertools import chain
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.model_latest_aliases import expand_model_latest_alias
from litellm.router_api import RegistryReader

LATEST_ALIAS_REQUESTED_MODEL_GROUP_METADATA_KEY: Final = "latest_alias_requested_model_group"


def underlying_models(deployments: Iterable[Mapping[str, object]] | None) -> Iterator[tuple[str, str]]:
    for deployment in deployments or ():
        name, params = deployment.get("model_name"), as_str_mapping(deployment.get("litellm_params"))
        underlying = None if params is None else params.get("model")
        if isinstance(name, str) and isinstance(underlying, str):
            yield name, underlying


def is_shared_deployment(deployment: Mapping[str, object]) -> bool:
    info: Final = as_str_mapping(deployment.get("model_info"))
    return info is None or not info.get("team_id")


def requested_latest_alias_model_group(metadata: Mapping[str, object], served: str) -> tuple[str, ...]:
    requested: Final = metadata.get(LATEST_ALIAS_REQUESTED_MODEL_GROUP_METADATA_KEY)
    return (requested,) if isinstance(requested, str) and requested != served else ()


def shared_underlying_models(deployments: Iterable[Mapping[str, object]] | None) -> Iterator[tuple[str, str]]:
    return underlying_models(filter(is_shared_deployment, deployments or ()))


def latest_alias_targets(names: Iterable[str]) -> frozenset[str]:
    return frozenset(chain.from_iterable(expand_model_latest_alias(name)[1:] for name in names))


def latest_alias_versioned_model_group(model: str, llm_router: RegistryReader) -> str | None:
    registry: Final = llm_router.registry()
    targets: Final = latest_alias_targets(name for _, name in underlying_models(registry.deployments(model)))
    if len(targets) != 1:
        return None
    return next(
        (
            name
            for name, underlying in shared_underlying_models(registry.deployments())
            if underlying in targets and name != model
        ),
        None,
    )


def expand_latest_alias_model_groups(model: str) -> tuple[str, ...]:
    from litellm.proxy.proxy_server import llm_router

    names: Final = expand_model_latest_alias(model)
    target: Final = None if llm_router is None else latest_alias_versioned_model_group(model, llm_router)
    return names if target is None or target in names else (*names, target)
