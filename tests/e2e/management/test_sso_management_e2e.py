"""Live e2e: the enterprise SSO configuration surface an admin drives when
onboarding an identity provider, for Microsoft Entra ID (Azure AD) and Okta.

This is the browser-free half of SSO: the OAuth login dance needs a real IdP, but
the configuration lifecycle an admin performs first - write the provider settings,
read them back, and confirm the proxy reports the provider ready - runs fully
against a live proxy. Each provider is its own class so the file reads as a spec
for how that provider is configured in production.

The SSO config is a singleton (one LiteLLM_SSOConfig row plus live os.environ), not
a per-test resource. A single PATCH /update/sso_settings is a full replace: the
endpoint clears every env var it isn't given, so configuring one provider isolates
it from the others. Every test that writes config defers a reset to the
unconfigured baseline, so the classes don't leak into each other or into sibling
suites. This assumes serial execution against one proxy process (the local proof
path); the settings are process-global, so a parallel run against a shared proxy
would race on them.

Requires STORE_MODEL_IN_DB=True on the proxy (the /update/sso_settings precondition)
and the master key. Not premium-gated: the SSO settings routes are admin-auth only,
unlike /sso/key/generate and the SCIM surface.
"""

from __future__ import annotations

import pytest

from e2e_http import NoBody, Result, UnauthorizedError, UnknownApiError
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import EntraSSOConfig, OktaSSOConfig, SSOSettingsResponse
from sso_management_client import SSOManagementClient

pytestmark = pytest.mark.e2e


def _configure(sso_client: SSOManagementClient, resources: ResourceManager, body: EntraSSOConfig | OktaSSOConfig) -> None:
    """Apply an SSO provider config and queue the reset first, so teardown clears
    the singleton even if the write half-applies or a later assertion fails."""
    resources.defer(sso_client.reset_sso_settings)
    sso_client.update_sso_settings(body)


def _assert_denied(route: str, result: Result[SSOSettingsResponse] | Result[NoBody]) -> None:
    """A non-admin key must be refused. LiteLLM denies an admin-only route by role
    with a 401 (unauthorized), and by an allowed_routes mismatch with a 403; either
    is a valid denial of the SSO settings surface, so accept both and reject any
    success or other outcome."""
    match result:
        case UnauthorizedError():
            return
        case UnknownApiError(status_code=403):
            return
        case _:
            pytest.fail(f"non-admin call to {route} must be denied (401 or 403), got {result}")


class TestEntraIdSSOManagement:
    @pytest.mark.covers("mgmt.sso_settings.update.persists")
    def test_update_persists_entra_provider_config(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        client_id = f"entra-client-{marker}"
        tenant = f"entra-tenant-{marker}"
        _configure(
            sso_client,
            resources,
            EntraSSOConfig(
                microsoft_client_id=client_id,
                microsoft_client_secret=f"entra-secret-{marker}-do-not-leak",
                microsoft_tenant=tenant,
            ),
        )

        values = sso_client.get_sso_settings()
        assert values.microsoft_client_id == client_id, (
            f"/get/sso_settings reports microsoft_client_id {values.microsoft_client_id!r}, configured {client_id!r}"
        )
        assert values.microsoft_tenant == tenant, (
            f"/get/sso_settings reports microsoft_tenant {values.microsoft_tenant!r}, configured {tenant!r}"
        )

    @pytest.mark.covers("mgmt.sso_settings.secret_masked")
    def test_client_secret_masked_on_read(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        secret = f"entra-secret-{marker}-DO-NOT-LEAK-plaintext"
        _configure(
            sso_client,
            resources,
            EntraSSOConfig(
                microsoft_client_id=f"entra-client-{marker}",
                microsoft_client_secret=secret,
                microsoft_tenant=f"entra-tenant-{marker}",
            ),
        )

        returned = sso_client.get_sso_settings().microsoft_client_secret
        assert returned is not None, "/get/sso_settings omitted microsoft_client_secret after it was configured"
        assert returned != secret, "/get/sso_settings returned the OAuth client secret verbatim (no masking)"
        assert secret not in returned, f"/get/sso_settings leaked the full client secret inside {returned!r}"
        assert marker not in returned, f"/get/sso_settings leaked the secret's distinctive middle inside {returned!r}"
        assert "*" in returned, f"/get/sso_settings did not mask the client secret; got {returned!r}"

    @pytest.mark.covers("mgmt.sso.readiness.reports_provider")
    def test_readiness_reports_microsoft_when_fully_configured(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        _configure(
            sso_client,
            resources,
            EntraSSOConfig(
                microsoft_client_id=f"entra-client-{marker}",
                microsoft_client_secret=f"entra-secret-{marker}",
                microsoft_tenant=f"entra-tenant-{marker}",
            ),
        )

        readiness = sso_client.sso_readiness()
        assert readiness.sso_configured is True, "/sso/readiness reports sso_configured false after Entra was configured"
        assert readiness.provider == "microsoft", (
            f"/sso/readiness reports provider {readiness.provider!r}, expected 'microsoft'"
        )
        assert readiness.status == "healthy", (
            f"/sso/readiness reports status {readiness.status!r} for a fully configured Entra provider"
        )

    @pytest.mark.covers("mgmt.sso.readiness.reports_missing_vars")
    def test_readiness_reports_missing_vars_when_partial(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        _configure(sso_client, resources, EntraSSOConfig(microsoft_client_id=f"entra-client-{marker}"))

        probe = sso_client.sso_readiness_probe()
        assert probe.status_code == 503, (
            f"/sso/readiness must 503 when Entra is configured without its secret/tenant, got "
            f"{probe.status_code}: {probe.body[:300]}"
        )
        assert "MICROSOFT_CLIENT_SECRET" in probe.body, (
            f"/sso/readiness 503 must name MICROSOFT_CLIENT_SECRET as missing, got: {probe.body[:300]}"
        )
        assert "MICROSOFT_TENANT" in probe.body, (
            f"/sso/readiness 503 must name MICROSOFT_TENANT as missing, got: {probe.body[:300]}"
        )


class TestOktaSSOManagement:
    @pytest.mark.covers("mgmt.sso_settings.update.persists")
    def test_update_persists_okta_provider_config(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        client_id = f"okta-client-{marker}"
        authorization_endpoint = f"https://e2e-{marker}.okta.com/oauth2/v1/authorize"
        _configure(
            sso_client,
            resources,
            OktaSSOConfig(
                generic_client_id=client_id,
                generic_client_secret=f"okta-secret-{marker}-do-not-leak",
                generic_authorization_endpoint=authorization_endpoint,
                generic_token_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/token",
                generic_userinfo_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/userinfo",
            ),
        )

        values = sso_client.get_sso_settings()
        assert values.generic_client_id == client_id, (
            f"/get/sso_settings reports generic_client_id {values.generic_client_id!r}, configured {client_id!r}"
        )
        assert values.generic_authorization_endpoint == authorization_endpoint, (
            f"/get/sso_settings reports generic_authorization_endpoint {values.generic_authorization_endpoint!r}, "
            f"configured {authorization_endpoint!r}"
        )

    @pytest.mark.covers("mgmt.sso_settings.secret_masked")
    def test_client_secret_masked_on_read(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        secret = f"okta-secret-{marker}-DO-NOT-LEAK-plaintext"
        _configure(
            sso_client,
            resources,
            OktaSSOConfig(
                generic_client_id=f"okta-client-{marker}",
                generic_client_secret=secret,
                generic_authorization_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/authorize",
                generic_token_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/token",
                generic_userinfo_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/userinfo",
            ),
        )

        returned = sso_client.get_sso_settings().generic_client_secret
        assert returned is not None, "/get/sso_settings omitted generic_client_secret after it was configured"
        assert returned != secret, "/get/sso_settings returned the OAuth client secret verbatim (no masking)"
        assert secret not in returned, f"/get/sso_settings leaked the full client secret inside {returned!r}"
        assert marker not in returned, f"/get/sso_settings leaked the secret's distinctive middle inside {returned!r}"
        assert "*" in returned, f"/get/sso_settings did not mask the client secret; got {returned!r}"

    @pytest.mark.covers("mgmt.sso.readiness.reports_provider")
    def test_readiness_reports_generic_when_fully_configured(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        _configure(
            sso_client,
            resources,
            OktaSSOConfig(
                generic_client_id=f"okta-client-{marker}",
                generic_client_secret=f"okta-secret-{marker}",
                generic_authorization_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/authorize",
                generic_token_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/token",
                generic_userinfo_endpoint=f"https://e2e-{marker}.okta.com/oauth2/v1/userinfo",
            ),
        )

        readiness = sso_client.sso_readiness()
        assert readiness.sso_configured is True, "/sso/readiness reports sso_configured false after Okta was configured"
        assert readiness.provider == "generic", (
            f"/sso/readiness reports provider {readiness.provider!r}, expected 'generic' (Okta is the generic provider)"
        )
        assert readiness.status == "healthy", (
            f"/sso/readiness reports status {readiness.status!r} for a fully configured Okta provider"
        )

    @pytest.mark.covers("mgmt.sso.readiness.reports_missing_vars")
    def test_readiness_reports_missing_vars_when_partial(
        self, sso_client: SSOManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        _configure(sso_client, resources, OktaSSOConfig(generic_client_id=f"okta-client-{marker}"))

        probe = sso_client.sso_readiness_probe()
        assert probe.status_code == 503, (
            f"/sso/readiness must 503 when Okta is configured without its secret/endpoints, got "
            f"{probe.status_code}: {probe.body[:300]}"
        )
        for expected in ("GENERIC_CLIENT_SECRET", "GENERIC_AUTHORIZATION_ENDPOINT", "GENERIC_TOKEN_ENDPOINT", "GENERIC_USERINFO_ENDPOINT"):
            assert expected in probe.body, (
                f"/sso/readiness 503 must name {expected} as missing, got: {probe.body[:300]}"
            )


class TestSSOSettingsAccessControl:
    @pytest.mark.covers("mgmt.sso_settings.update.admin_only")
    def test_non_admin_key_forbidden_from_sso_settings(
        self, sso_client: SSOManagementClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """A non-admin virtual key can neither read nor write the SSO provider config
        (an admin-only surface), and a rejected write persists nothing. The reset is
        deferred defensively: if the write were wrongly accepted, teardown still
        clears it."""
        resources.defer(sso_client.reset_sso_settings)

        _assert_denied("/get/sso_settings", sso_client.get_sso_settings_as(scoped_key))

        forbidden_client_id = f"should-not-persist-{unique_marker()}"
        _assert_denied(
            "/update/sso_settings",
            sso_client.update_sso_settings_as(scoped_key, EntraSSOConfig(microsoft_client_id=forbidden_client_id)),
        )

        assert sso_client.get_sso_settings().microsoft_client_id != forbidden_client_id, (
            "a non-admin PATCH that was supposed to be forbidden still persisted microsoft_client_id"
        )
