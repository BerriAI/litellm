from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.types.router import LiteLLM_Params


def deployment_sets_litellm_param(deployment_params: Mapping[str, object], key: str) -> bool:
    value: Final = deployment_params.get(key)
    if value is None:
        return False
    field: Final = LiteLLM_Params.model_fields.get(key)
    return field is None or value != field.default


def router_defaults_for_deployment(
    router_defaults: Mapping[str, object], deployment_params: Mapping[str, object]
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            key: value
            for key, value in router_defaults.items()
            if value is not None and not deployment_sets_litellm_param(deployment_params, key)
        }
    )


def params_sent_to_deployment(
    deployment_params: Mapping[str, object], router_defaults: Mapping[str, object], request: Mapping[str, object]
) -> Mapping[str, object]:
    return MappingProxyType(
        {**deployment_params, **router_defaults_for_deployment(router_defaults, deployment_params), **request}
    )
