from collections.abc import Mapping
from typing import Final

from litellm.rust_bridge.host.routes.tokenize import body_params


def deployment_kwargs(params: Mapping[str, object]) -> Mapping[str, object]:
    provider_params: Final = body_params(params)
    return {
        **{name: value for name, value in params.items() if name not in provider_params},
        "_tokenize_deployment_params": provider_params,
    }
