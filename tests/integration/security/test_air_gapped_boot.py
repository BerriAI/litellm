from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from integration._support.client import Gateway, string_value
from integration._support.forward_proxy import refusing_forward_proxy
from integration._support.process import owned_proxy

_BUNDLED_COST_MAP: Final = Path(__file__).resolve().parents[3] / "litellm/model_prices_and_context_window_backup.json"


def _git_blob_id(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


def test_proxy_becomes_ready_without_network(gateway: Gateway, tmp_path: Path) -> None:
    with refusing_forward_proxy() as egress:
        no_network: Final = {
            "HTTPS_PROXY": egress.url,
            "HTTP_PROXY": egress.url,
            "ALL_PROXY": egress.url,
            "NO_PROXY": "127.0.0.1,localhost",
        }
        with owned_proxy(
            gateway, tmp_path, no_network, remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_LICENSE")
        ) as isolated:
            readiness: Final = isolated.request("GET", "/health/readiness")
            source: Final = isolated.get("/model/cost_map/source")
        refused: Final = frozenset(egress.targets())

    assert readiness.status_code == 200, readiness.text
    assert source["source"] == "local"
    assert source["is_env_forced"] is False
    assert string_value(source["fallback_reason"]).startswith("Remote fetch failed")
    assert source["source_revision"] == _git_blob_id(_BUNDLED_COST_MAP.read_bytes())
    assert refused == {f"{urlsplit(string_value(source['url'])).hostname}:443"}
