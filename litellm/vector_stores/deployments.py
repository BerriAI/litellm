from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Final  # noqa: TID251  # load_config indexes and rewrites each model_list entry in place

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

VECTOR_STORE_DEPLOYMENT_MODE: Final = "vector_store"
VECTOR_STORE_DEPLOYMENT_PARAM: Final = "vector_store_deployment"
_STORE_ONLY_CONNECTION_PARAMS: Final = frozenset({"api_base", "api_key", "litellm_credential_name"})
_GROUP_ROUTING_PARAMS: Final = frozenset({"custom_llm_provider", "order"})


class VectorStoreDeploymentParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    custom_llm_provider: str

    @model_validator(mode="after")
    def _model_is_derived(self) -> "VectorStoreDeploymentParams":
        if "model" in (self.model_extra or {}):
            raise ValueError("litellm_params.model is not used for vector store deployments. Remove it.")
        return self


class VectorStoreDeploymentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    deployment_name: str
    litellm_params: VectorStoreDeploymentParams
    model_info: Mapping[str, object] = Field(default_factory=dict)


_DEPLOYMENTS: Final = TypeAdapter(tuple[VectorStoreDeploymentConfig, ...])


def parse_vector_store_deployments(config: Mapping[str, object]) -> tuple[VectorStoreDeploymentConfig, ...]:
    return _DEPLOYMENTS.validate_python(config.get("vector_store_deployments") or ())


def router_deployment(
    deployment: VectorStoreDeploymentConfig,
) -> dict[str, Any]:  # mutable-ok: the model_list entry load_config resolves secrets into
    params: Final = deployment.litellm_params
    return {
        "model_name": deployment.deployment_name,
        "litellm_params": {
            **params.model_dump(),
            "model": f"{params.custom_llm_provider}/{deployment.deployment_name}",
        },
        "model_info": {**deployment.model_info, "mode": VECTOR_STORE_DEPLOYMENT_MODE},
    }


def config_model_list(
    config: Mapping[str, object],
) -> list[Any]:  # mutable-ok: load_config resolves os.environ secrets into each entry in place
    """The config's model_list plus its vector store deployments, as the router and the DB model sync see them."""
    raw: Final = config.get("model_list")
    model_list: Final = as_object_list(raw) if raw is not None else ()
    if model_list is None:
        raise TypeError("model_list must be a list")
    return [*model_list, *(router_deployment(deployment) for deployment in parse_vector_store_deployments(config))]


def is_vector_store_deployment(model_info: object) -> bool:
    info: Final = as_str_mapping(model_info)
    mode: Final[object] = info.get("mode") if info is not None else getattr(model_info, "mode", None)
    return mode == VECTOR_STORE_DEPLOYMENT_MODE


def is_vector_store_router_deployment(deployment: Mapping[str, object]) -> bool:
    return is_vector_store_deployment(deployment.get("model_info"))


def _store_problem(
    store_name: object,
    store_params: Mapping[str, object],
    group: str,
    deployments: tuple[VectorStoreDeploymentConfig, ...],
) -> str | None:
    if not deployments:
        return f"vector store '{store_name}' references vector_store_deployment '{group}', which is not defined"
    providers: Final = frozenset(deployment.litellm_params.custom_llm_provider for deployment in deployments)
    store_provider: Final = store_params.get("custom_llm_provider")
    if store_provider not in providers:
        return (
            f"vector store '{store_name}' has custom_llm_provider '{store_provider}' but vector_store_deployment "
            f"'{group}' serves {sorted(providers)}"
        )
    deployment_keys: Final[frozenset[str]] = frozenset().union(
        *(deployment.litellm_params.model_dump().keys() for deployment in deployments)
    )
    overlap: Final = (_STORE_ONLY_CONNECTION_PARAMS | (deployment_keys - _GROUP_ROUTING_PARAMS)) & store_params.keys()
    if overlap:
        return (
            f"vector store '{store_name}' uses vector_store_deployment '{group}' and also sets {sorted(overlap)}. "
            "Connection settings belong on the vector_store_deployments entries."
        )
    return None


def _grouped_store_problem(store: object, deployments: tuple[VectorStoreDeploymentConfig, ...]) -> str | None:
    store_fields: Final = as_str_mapping(store) or {}
    params: Final = as_str_mapping(store_fields.get("litellm_params"))
    group: Final = params.get(VECTOR_STORE_DEPLOYMENT_PARAM) if params is not None else None
    if params is None or group is None:
        return None
    store_name: Final = store_fields.get("vector_store_name")
    if not isinstance(group, str):
        return f"vector store '{store_name}' has a non-string vector_store_deployment"
    return _store_problem(
        store_name,
        params,
        group,
        tuple(deployment for deployment in deployments if deployment.deployment_name == group),
    )


def vector_store_deployment_problems(config: Mapping[str, object]) -> tuple[str, ...]:
    """Why the config's vector_store_deployments and the stores that use them cannot serve, if they cannot."""
    deployments: Final = parse_vector_store_deployments(config)
    model_names: Final = frozenset(
        str((as_str_mapping(model) or {}).get("model_name")) for model in as_object_list(config.get("model_list")) or ()
    )
    clashing: Final = sorted(frozenset(deployment.deployment_name for deployment in deployments) & model_names)
    clash: Final = (
        (f"vector_store_deployments names {clashing} are already model_list model names",) if clashing else ()
    )
    stores: Final = as_object_list(config.get("vector_store_registry")) or ()
    return clash + tuple(
        problem for store in stores if (problem := _grouped_store_problem(store, deployments)) is not None
    )


def _order(deployment: Mapping[str, object]) -> float:
    params: Final = as_str_mapping(deployment.get("litellm_params")) or {}
    order: Final = params.get("order")
    return float(order) if isinstance(order, int) else float("inf")


def with_primary_deployment(
    store_params: Mapping[str, object], deployments: Iterable[Mapping[str, object]]
) -> Mapping[str, object]:
    """A grouped store's params on its group's lowest-order deployment, for paths that cannot fail over."""
    group: Final = store_params.get(VECTOR_STORE_DEPLOYMENT_PARAM)
    if not isinstance(group, str):
        return store_params
    members: Final = sorted(
        (deployment for deployment in deployments if deployment.get("model_name") == group), key=_order
    )
    if not members:
        raise ValueError(f"vector_store_deployment '{group}' has no deployments")
    primary_params: Final = as_str_mapping(members[0].get("litellm_params")) or {}
    return MappingProxyType(
        {
            **{key: value for key, value in store_params.items() if key != VECTOR_STORE_DEPLOYMENT_PARAM},
            **{key: value for key, value in primary_params.items() if key not in ("model", "order")},
        }
    )


def with_deployment_group(store_params: Mapping[str, object]) -> Mapping[str, object]:
    """A grouped store's request params, routed to its deployment group in place of any caller-chosen model."""
    group: Final = store_params.get(VECTOR_STORE_DEPLOYMENT_PARAM)
    if not isinstance(group, str):
        return store_params
    return MappingProxyType(
        {
            **{key: value for key, value in store_params.items() if key != VECTOR_STORE_DEPLOYMENT_PARAM},
            "model": group,
        }
    )
