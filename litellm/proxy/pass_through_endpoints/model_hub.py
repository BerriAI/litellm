"""Pass-through endpoints published on the public Model Hub."""

from typing import Final

from litellm.proxy._types import PassThroughGenericEndpoint
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import configured_pass_through_endpoints
from litellm.types.proxy.management_endpoints.model_management_endpoints import ModelGroupInfoProxy

PASS_THROUGH_MODE: Final = "passthrough"


def pass_through_model_hub_row(endpoint: PassThroughGenericEndpoint) -> ModelGroupInfoProxy:
    return ModelGroupInfoProxy(
        model_group=endpoint.display_name or endpoint.path,
        providers=[],
        mode=PASS_THROUGH_MODE,
        is_public_model_group=True,
        pass_through_path=endpoint.path,
        pass_through_methods=endpoint.methods,
    )


async def published_pass_through_rows() -> tuple[ModelGroupInfoProxy, ...]:
    endpoints: Final = await configured_pass_through_endpoints()
    return tuple(pass_through_model_hub_row(endpoint) for endpoint in endpoints if endpoint.show_in_model_hub)
