"""Live e2e: the process-lifecycle probes Kubernetes and load balancers depend on.

Liveness and public readiness must answer without a credential (a load balancer
has none), and public readiness must distinguish a healthy worker from one whose
DB is unreachable by reporting the resolved DB state. The detailed readiness
route, by contrast, is authenticated: it exposes diagnostics (version, callbacks,
DB) and must reject an anonymous caller. The suite runs against a proxy configured
with a real database, so a healthy readiness payload reports the DB as connected;
a regression that stopped checking the DB, or dropped the public exposure, fails
here.
"""

from __future__ import annotations

import pytest

from e2e_config import MASTER_KEY
from e2e_http import UnauthorizedError, unwrap
from other_client import OtherClient

pytestmark = pytest.mark.e2e


class TestHealthLifecycle:
    @pytest.mark.covers("other.lifecycle.liveness.ping")
    def test_liveness_reports_alive_without_auth(self, client: OtherClient) -> None:
        probe = client.liveness()
        assert probe.status_code == 200, (
            f"liveness must answer 200 for an unauthenticated probe, got "
            f"{probe.status_code}: {probe.body[:200]}"
        )
        assert "alive" in probe.body.lower(), (
            f"liveness body must confirm the worker is alive, got {probe.body[:200]}"
        )

    @pytest.mark.covers("other.lifecycle.readiness.public_probe")
    def test_readiness_is_reachable_without_credentials(self, client: OtherClient) -> None:
        readiness = unwrap(client.readiness_public())
        assert readiness.status == "healthy", (
            f"public readiness must report a healthy worker, got status {readiness.status!r}"
        )

    @pytest.mark.covers("other.lifecycle.readiness.reports_db_status")
    def test_readiness_reports_connected_db(self, client: OtherClient) -> None:
        readiness = unwrap(client.readiness_public())
        assert readiness.db == "connected", (
            "readiness must report the configured database as connected so an "
            f"orchestrator can tell a healthy worker from a DB-unreachable one, got {readiness.db!r}"
        )

    @pytest.mark.covers("other.lifecycle.readiness_details.authenticated_diagnostics")
    def test_readiness_details_require_auth_and_expose_diagnostics(self, client: OtherClient) -> None:
        anonymous = client.readiness_details_unauthenticated()
        assert isinstance(anonymous, UnauthorizedError), (
            f"/health/readiness/details must reject an unauthenticated caller, got {anonymous}"
        )

        details = unwrap(client.readiness_details(MASTER_KEY))
        assert details.status == "healthy", f"authenticated readiness status must be healthy, got {details.status!r}"
        assert details.litellm_version is not None, (
            "authenticated diagnostics must expose the litellm version"
        )
        assert details.db == "connected", (
            f"authenticated diagnostics must report the DB as connected, got {details.db!r}"
        )
