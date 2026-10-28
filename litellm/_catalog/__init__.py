"""Internal catalog API backed by the Rust ``model-catalog`` crate. Not public, its shape may change in any release"""

import os
import sys
from typing import Final

from litellm._logging import verbose_logger
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
    Refresher,
    Registered,
    Remote,
    Resolved,
    RowError,
    Unavailable,
    overlay,
)

Source = Remote | File | Bundled

CATALOG: Final = Catalog.bound_to(sys.modules["litellm"], "model_cost")

_REPORTED_INVALID_ROWS: Final[set[str]] = set()  # mutable-ok: logs each bad live row once per process


def report_invalid(invalid: Invalid) -> None:
    """Logs a row that fails validation, once per key per process (decision 20)"""
    if invalid.key in _REPORTED_INVALID_ROWS:
        return
    _REPORTED_INVALID_ROWS.add(invalid.key)
    verbose_logger.warning("litellm.model_cost[%r] fails validation: %s", invalid.key, "; ".join(invalid.errors))


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
    "Refresher",
    "Registered",
    "Remote",
    "Resolved",
    "RowError",
    "Source",
    "Unavailable",
    "default_source",
    "overlay",
    "report_invalid",
]
