"""Internal catalog API backed by the Rust ``model-catalog`` crate. Not public, its shape may change in any release"""

import sys
from typing import Final

from litellm.rust_bridge._native import Catalog, Fetched, Installed, RowError, Unavailable

CATALOG: Final = Catalog.bound_to(sys.modules["litellm"], "model_cost")

__all__ = ["CATALOG", "Catalog", "Fetched", "Installed", "RowError", "Unavailable"]
