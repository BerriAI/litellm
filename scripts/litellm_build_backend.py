"""PEP 517 backend that builds a pure-Python wheel when LITELLM_SKIP_RUST_BRIDGE is set."""

import os
from typing import Final

SKIP_RUST_BRIDGE: Final = os.environ.get("LITELLM_SKIP_RUST_BRIDGE", "").lower() in {"1", "true"}

if SKIP_RUST_BRIDGE:
    from hatchling.build import (
        build_editable,
        build_sdist,
        build_wheel,
        get_requires_for_build_editable,
        get_requires_for_build_sdist,
        get_requires_for_build_wheel,
        prepare_metadata_for_build_editable,
        prepare_metadata_for_build_wheel,
    )
else:
    from maturin import (
        build_editable,
        build_sdist,
        build_wheel,
        get_requires_for_build_editable,
        get_requires_for_build_sdist,
        get_requires_for_build_wheel,
        prepare_metadata_for_build_editable,
        prepare_metadata_for_build_wheel,
    )

__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]
