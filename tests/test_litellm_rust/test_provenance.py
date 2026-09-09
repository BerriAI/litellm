from typing import Final

import pytest

from litellm.rust_bridge.loader import get_native_bridge

pytestmark = pytest.mark.requires_rust_extension


def test_native_route_catalogue_matches_extension_exports() -> None:
    from litellm.rust_bridge.provenance import (
        RUST_NON_RESPONSE_ENTRYPOINTS,
        RUST_RESPONSE_ENTRYPOINTS,
    )

    native: Final = get_native_bridge()
    assert native is not None
    route_exports: Final = frozenset(name for name in dir(native) if not name.startswith("_") and name[0].islower())

    assert route_exports == RUST_RESPONSE_ENTRYPOINTS | RUST_NON_RESPONSE_ENTRYPOINTS
