"""Pass-through endpoints published on the public Model Hub."""

from collections.abc import Iterator
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import PassThroughGenericEndpoint
from litellm.types.proxy.management_endpoints.model_management_endpoints import ModelGroupInfoProxy

PASS_THROUGH_MODE: Final = "passthrough"
_SERVING_ENTRIES: Final = TypeAdapter(list[dict[str, object]])


def pass_through_model_hub_row(endpoint: PassThroughGenericEndpoint) -> ModelGroupInfoProxy:
    return ModelGroupInfoProxy(
        model_group=endpoint.display_name or endpoint.path,
        providers=[],
        mode=PASS_THROUGH_MODE,
        is_public_model_group=True,
        pass_through_path=endpoint.path,
        pass_through_methods=endpoint.methods,
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


def published_pass_through_rows() -> tuple[ModelGroupInfoProxy, ...]:
    return tuple(
        pass_through_model_hub_row(endpoint)
        for endpoint in _serving_pass_through_endpoints()
        if endpoint.show_in_model_hub
    )
