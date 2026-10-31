"""Saving SSO settings back the way the dashboard does must keep the stored client secret."""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import PROXY_BASE_URL, unique_marker
from e2e_metadata import Domain, Route, Subject, meta
from idp import Keycloak
from lifecycle import ResourceManager
from management.management_client import ManagementClient
from models import CLEAR, SSORoleMappings, SSOSettingsUpdateBody

pytestmark = pytest.mark.e2e


def _cleared_sso_settings() -> SSOSettingsUpdateBody:
    return SSOSettingsUpdateBody(
        generic_client_id=CLEAR,
        generic_client_secret=CLEAR,
        generic_authorization_endpoint=CLEAR,
        generic_token_endpoint=CLEAR,
        generic_userinfo_endpoint=CLEAR,
        proxy_base_url=CLEAR,
        role_mappings=CLEAR,
    )


class TestSsoSettingsUpdate:
    @pytest.mark.covers("mgmt.sso_settings.update.secret_preserved")
    @meta(Subject(domain=Domain.PROXY_AUTH, route=Route.ADMIN_UI))
    def test_saving_role_mappings_keeps_the_client_secret(
        self, client: ManagementClient, idp: Keycloak, resources: ResourceManager
    ) -> None:
        """The IdP is the judge: a bogus authorization code exchanged with the right client
        secret is logged as invalid_code, one exchanged with a wrong secret as
        invalid_client_credentials. The dashboard only ever holds the masked secret, so the
        second save sends the mask back."""
        browser: Final = idp.browser_client(callback_url=f"{PROXY_BASE_URL}/sso/callback", defer=resources.defer)
        idp.enable_login_events()
        discovery: Final = idp.discovery()
        resources.defer(lambda: client.update_sso_settings(_cleared_sso_settings()))
        client.update_sso_settings(
            SSOSettingsUpdateBody(
                generic_client_id=browser.client_id,
                generic_client_secret=browser.secret,
                generic_authorization_endpoint=discovery.authorization_endpoint,
                generic_token_endpoint=discovery.token_endpoint,
                generic_userinfo_endpoint=discovery.userinfo_endpoint,
                proxy_base_url=PROXY_BASE_URL,
            )
        )
        client.sso_callback(code=f"e2e-bogus-code-{unique_marker()}")
        assert idp.code_to_token_errors(browser) == ("invalid_code",), (
            "the freshly saved secret must authenticate the proxy at the IdP before the round trip proves anything"
        )

        shown: Final = client.sso_settings().values
        assert shown.generic_client_secret != browser.secret, "/get/sso_settings must not show the secret in clear"
        saved: Final = client.update_sso_settings(
            SSOSettingsUpdateBody(
                generic_client_id=shown.generic_client_id,
                generic_client_secret=shown.generic_client_secret,
                generic_authorization_endpoint=shown.generic_authorization_endpoint,
                generic_token_endpoint=shown.generic_token_endpoint,
                generic_userinfo_endpoint=shown.generic_userinfo_endpoint,
                generic_scope=shown.generic_scope,
                proxy_base_url=shown.proxy_base_url,
                role_mappings=SSORoleMappings(
                    provider="generic", group_claim="groups", roles={"proxy_admin": [f"e2e-admins-{unique_marker()}"]}
                ),
            )
        )
        assert saved.settings.generic_client_secret != browser.secret
        assert saved.settings.role_mappings is not None

        client.sso_callback(code=f"e2e-bogus-code-{unique_marker()}")
        assert idp.code_to_token_errors(browser) == ("invalid_code", "invalid_code"), (
            "the IdP rejected the proxy's client credentials: saving the settings page replaced the stored secret"
        )
