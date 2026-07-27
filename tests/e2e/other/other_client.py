"""Client for the `other` holding-pen suite: the auth gate (master key vs an
invalid key on an admin route) and the process-lifecycle health probes
(liveness, public readiness, authenticated readiness diagnostics).

Holds the shared ProxyClient so `resources` / `scoped_key` still clean up, and
adds only the routes these behaviors need. The health probes deliberately send
no auth header (public routes), so they go through the transport with an empty
headers model rather than a bearer.
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_http import NoBody, ProbeResult, Result
from models import (
    ReadinessDetailsResponse,
    ReadinessResponse,
    UserListParams,
    UserListResponse,
)
from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class OtherClient:
    proxy: ProxyClient

    def liveness(self) -> ProbeResult:
        """GET /health/liveliness. Unauthenticated; the probe returns status +
        raw body so the test can assert the worker reports itself alive."""
        return self.proxy.transport.probe("/health/liveliness", params=NoBody())

    def readiness_public(self) -> Result[ReadinessResponse]:
        """GET /health/readiness with no credential at all, proving the probe is
        safe to expose to an unauthenticated load balancer."""
        return self.proxy.transport.get(
            "/health/readiness",
            headers=NoBody(),
            params=NoBody(),
            response_type=ReadinessResponse,
        )

    def readiness_details(self, key: str) -> Result[ReadinessDetailsResponse]:
        return self.proxy.transport.get(
            "/health/readiness/details",
            headers=self.proxy.transport.bearer(key),
            params=NoBody(),
            response_type=ReadinessDetailsResponse,
        )

    def readiness_details_unauthenticated(self) -> Result[ReadinessDetailsResponse]:
        return self.proxy.transport.get(
            "/health/readiness/details",
            headers=NoBody(),
            params=NoBody(),
            response_type=ReadinessDetailsResponse,
        )

    def list_users_as(self, key: str) -> Result[UserListResponse]:
        """GET /user/list under `key`. Admin-only, so it doubles as the master
        key's authorization proof: the master key (proxy admin) reads it, a
        non-matching key is rejected before it ever reaches the handler."""
        return self.proxy.transport.get(
            "/user/list",
            headers=self.proxy.transport.bearer(key),
            params=UserListParams(user_ids="e2e-test-user"),
            response_type=UserListResponse,
        )


def build_client(proxy: ProxyClient) -> OtherClient:
    return OtherClient(proxy=proxy)
