"""Internal catalog API backed by the Rust ``model-catalog`` crate. Not public, its shape may change in any release"""

import os
import sys
from typing import Final

from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import (
    Bundled,
    Catalog,
    Fetched,
    File,
    Installed,
    Invalid,
    Missing,
    Provenance,
    Registered,
    Remote,
    Resolved,
    RowError,
    Unavailable,
    overlay,
)

Source = Remote | File | Bundled

CATALOG: Final = Catalog.bound_to(sys.modules["litellm"], "model_cost")


def default_source() -> Source:
    """Main's reload rule, the bundled map under offline mode or ``LITELLM_LOCAL_MODEL_COST_MAP``"""
    import litellm

    if is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        return Bundled()
    return Remote(litellm.model_cost_map_url)


__all__ = [
    "CATALOG",
    "Bundled",
    "Catalog",
    "Fetched",
    "File",
    "Installed",
    "Invalid",
    "Missing",
    "Provenance",
    "Registered",
    "Remote",
    "Resolved",
    "RowError",
    "Source",
    "Unavailable",
    "default_source",
    "overlay",
]
