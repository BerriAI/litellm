from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.proxy.auth.auth_utils import _BANNED_REQUEST_BODY_PARAMS
from litellm.rust_bridge import _native

pytestmark = pytest.mark.requires_rust_extension

_REGISTRY: Final = TypeAdapter(list[dict[str, object]])
_STEERING_CLASSES: Final = frozenset({"destination", "identity", "audience"})


def test_every_rust_param_that_steers_credentials_is_banned_from_proxy_bodies() -> None:
    registry: Final = _REGISTRY.validate_python(_native.control_params())
    steering: Final = frozenset(str(entry["name"]) for entry in registry if entry["class"] in _STEERING_CLASSES)
    assert "api_base" in steering
    assert steering <= frozenset(_BANNED_REQUEST_BODY_PARAMS), sorted(steering - frozenset(_BANNED_REQUEST_BODY_PARAMS))
