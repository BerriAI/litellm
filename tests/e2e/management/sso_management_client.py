"""Client for the SSO-management e2e suite: the shared ProxyClient plus the
enterprise SSO configuration surface an admin drives without a browser OAuth
dance - the /update/sso_settings write, the /get/sso_settings read-back (with
its masked secrets), and the /sso/readiness health check that reflects the
configured provider.

Kept separate from ManagementClient so the key/team/user routes and the SSO
routes stay independent; both hold the same shared ProxyClient, so the resources
fixture tears down through it either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_http import NoBody, ProbeResult, Result, unwrap
from models import (
    SSOConfigBody,
    SSOConfigClear,
    SSOReadinessResponse,
    SSOSettingsResponse,
    SSOSettingsValues,
)
from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class SSOManagementClient:
    proxy: ProxyClient

    def update_sso_settings(self, body: SSOConfigBody) -> None:
        _ = unwrap(
            self.proxy.transport.patch(
                "/update/sso_settings",
                headers=self.proxy.transport.master,
                json=body,
                response_type=NoBody,
            )
        )

    def reset_sso_settings(self) -> None:
        """Clear every SSO env var + the stored config back to the unconfigured
        baseline. Best-effort (teardown), so a failure is swallowed rather than
        raised."""
        _ = self.proxy.transport.patch(
            "/update/sso_settings",
            headers=self.proxy.transport.master,
            json=SSOConfigClear(),
            response_type=NoBody,
        )

    def get_sso_settings(self) -> SSOSettingsValues:
        return unwrap(
            self.proxy.transport.get(
                "/get/sso_settings",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=SSOSettingsResponse,
            )
        ).values

    def get_sso_settings_as(self, key: str) -> Result[SSOSettingsResponse]:
        """GET /get/sso_settings under an arbitrary key, so the access-control test
        can assert a non-admin key is refused (a 403 comes back as UnknownApiError)."""
        return self.proxy.transport.get(
            "/get/sso_settings",
            headers=self.proxy.transport.bearer(key),
            params=NoBody(),
            response_type=SSOSettingsResponse,
        )

    def update_sso_settings_as(self, key: str, body: SSOConfigBody) -> Result[NoBody]:
        """PATCH /update/sso_settings under an arbitrary key, for the same
        access-control assertion on the write path."""
        return self.proxy.transport.patch(
            "/update/sso_settings",
            headers=self.proxy.transport.bearer(key),
            json=body,
            response_type=NoBody,
        )

    def sso_readiness(self) -> SSOReadinessResponse:
        return unwrap(
            self.proxy.transport.get(
                "/sso/readiness",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=SSOReadinessResponse,
            )
        )

    def sso_readiness_probe(self) -> ProbeResult:
        """Raw /sso/readiness outcome, for the partial-config path where the proxy
        answers 503 (a non-2xx that would fail the typed read-back)."""
        return self.proxy.transport.probe("/sso/readiness", params=NoBody())


def build_sso_client(proxy: ProxyClient) -> SSOManagementClient:
    return SSOManagementClient(proxy=proxy)
