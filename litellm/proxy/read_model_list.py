"""Resolve a proxy config's ``model_list`` for the Rust AI gateway.

The Rust gateway calls this once at load time (via an embedded interpreter) and
builds its own (Rust) router from the returned ``model_list``. We do NOT call
``ProxyConfig.load_config`` here: that returns a *Python* ``litellm.Router`` (not
usable from Rust) and boots the whole proxy (callbacks, cache, DB, auth) as side
effects.

Instead we reuse ``ProxyConfig.get_config`` — the actual config reader — so the
gateway inherits the same heavy lifting the proxy does: ``include:`` merging,
``os.environ/`` + secret-manager resolution, and DB-stored models (when a DB is
configured). It has no proxy-setup side effects. Returns the resolved
``model_list``; the Rust side deserializes each entry into its ``Deployment``.
"""

from __future__ import annotations

import asyncio
from typing import Any


def read_model_list(config_path: str) -> list[dict[str, Any]]:
    """Load ``config_path`` via the proxy's own reader and return its
    resolved ``model_list``."""
    from litellm.proxy.proxy_server import ProxyConfig

    config = asyncio.run(ProxyConfig().get_config(config_file_path=config_path))
    return config.get("model_list") or []
