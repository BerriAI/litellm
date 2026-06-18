"""Spend-tracking e2e client: the generic ProxyClient plus spend-specific reads.

Generic proxy operations (keys, customers, chat/embed, route probing, SpendLogs
polling) live in the shared tests/e2e_tests/proxy_client.py. This module adds only
the spend-specific endpoints: /spend/calculate, /spend/tags, and key-spend polling.

Re-exports CallResult / SpendLogRow / require_successful_call / unique_marker so
existing tests keep importing them from here.
"""

import time
from typing import List, Optional

import requests

from proxy_client import (
    CallResult,
    ProbeResult,
    ProxyClient,
    SpendLogRow,
    _auth,
    proxy_client_kwargs,
    require_successful_call,
    unique_marker,
)

__all__ = [
    "SpendE2EClient",
    "build_client",
    "CallResult",
    "ProbeResult",
    "SpendLogRow",
    "require_successful_call",
    "unique_marker",
]


class SpendE2EClient(ProxyClient):
    def calculate_spend(self, model: str, content: str) -> float:
        resp = requests.post(
            f"{self._base_url}/spend/calculate",
            headers=_auth(self._master_key),
            json={"model": model, "messages": [{"role": "user", "content": content}]},
            timeout=self._request_timeout,
        )
        resp.raise_for_status()
        return float(resp.json()["cost"])

    def spend_by_tags(self) -> List[SpendLogRow]:
        """/spend/tags: SUM(spend) GROUP BY tag, straight from SpendLogs."""
        resp = requests.get(
            f"{self._base_url}/spend/tags",
            headers=_auth(self._master_key),
            timeout=self._request_timeout,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        if isinstance(data, dict) and "spend_per_tag" in data:
            data = data["spend_per_tag"]
        return [dict(row) for row in data] if isinstance(data, list) else []

    def poll_tag_spend(
        self, tag: str, *, minimum: float = 0.0
    ) -> Optional[SpendLogRow]:
        """Poll until the tag's aggregate spend reaches `minimum`; last seen entry."""
        deadline = time.monotonic() + self._poll_timeout
        entry: Optional[SpendLogRow] = None
        while time.monotonic() < deadline:
            matches = [
                row
                for row in self.spend_by_tags()
                if str(row.get("individual_request_tag")) == tag
            ]
            if matches:
                entry = matches[0]
                if float(entry.get("total_spend") or 0.0) >= minimum:
                    return entry
            time.sleep(self._poll_interval)
        return entry

    def poll_key_spend(self, key: str, *, minimum: float = 0.0) -> float:
        deadline = time.monotonic() + self._poll_timeout
        spend = 0.0
        while time.monotonic() < deadline:
            spend = float(self.key_info(key).get("spend") or 0.0)
            if spend > minimum:
                return spend
            time.sleep(self._poll_interval)
        return spend


def build_client() -> SpendE2EClient:
    return SpendE2EClient(**proxy_client_kwargs())
