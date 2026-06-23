"""Resolve a proxy config's ``model_list`` for the Rust AI gateway.

The Rust gateway calls this once at load time (via an embedded interpreter) so it
never has to reimplement ``os.environ/`` / secret-manager resolution — that logic
lives in ``ProxyConfig._check_for_os_environ_vars`` and is reused here verbatim.

Returns the fully-resolved ``model_list`` (a list of deployment dicts). The Rust
side deserializes each entry into its ``Deployment`` type.
"""

from __future__ import annotations

from typing import Any

import yaml


def read_model_list(config_path: str) -> list[dict[str, Any]]:
    """Load ``config_path`` and return its resolved ``model_list``.

    ``os.environ/...`` markers and secret-manager references are resolved using
    the proxy's own reader, so the result matches what the proxy itself would use.
    """
    # Imported lazily: pulling in proxy_server at module load is heavy and risks
    # circular imports; this helper is only ever called once at gateway startup.
    from litellm.proxy.proxy_server import ProxyConfig

    with open(config_path) as config_file:
        config: Any = yaml.safe_load(config_file) or {}

    config = ProxyConfig()._check_for_os_environ_vars(config=config)
    return config.get("model_list") or []
