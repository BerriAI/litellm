import pytest

from litellm.proxy.management_endpoints.sso.id_jag_assertion_capture import (
    ActiveSSOProvider,
    active_sso_provider,
    id_jag_assertion_capture_gap,
    id_jag_assertion_capture_gap_at_startup,
)

_SSO_ENV_VARS = (
    "GOOGLE_CLIENT_ID",
    "MICROSOFT_CLIENT_ID",
    "GENERIC_CLIENT_ID",
    "SAML_IDP_METADATA_URL",
    "SAML_IDP_METADATA_XML",
)


@pytest.fixture(autouse=True)
def _isolated_sso_env(monkeypatch):
    """Every SSO selector is read from the process environment, so a value left behind by
    another test would silently decide this one's answer."""
    for name in _SSO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestActiveSSOProviderMirrorsTheCallback:
    """The gap warning is only as good as its agreement with the branch the login callback
    actually takes, so provider selection is asserted branch by branch, including the
    precedence that makes a co-configured generic client unreachable."""

    def test_google_client_id_selects_google(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-cid")
        assert active_sso_provider() is ActiveSSOProvider.google

    def test_microsoft_client_id_selects_microsoft(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_CLIENT_ID", "ms-cid")
        assert active_sso_provider() is ActiveSSOProvider.microsoft

    def test_generic_client_id_selects_generic(self, monkeypatch):
        monkeypatch.setenv("GENERIC_CLIENT_ID", "generic-cid")
        assert active_sso_provider() is ActiveSSOProvider.generic

    def test_saml_metadata_selects_saml(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_METADATA_URL", "https://idp.example.com/metadata")
        assert active_sso_provider() is ActiveSSOProvider.saml

    def test_nothing_configured_selects_none(self):
        assert active_sso_provider() is ActiveSSOProvider.none

    def test_google_outranks_a_co_configured_generic_client(self, monkeypatch):
        """The callback tests GOOGLE_CLIENT_ID first, so the generic arm never runs here and the
        assertion the refresher later renews was minted by the Google client, not the generic one."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-cid")
        monkeypatch.setenv("GENERIC_CLIENT_ID", "generic-cid")
        assert active_sso_provider() is ActiveSSOProvider.google

    def test_microsoft_outranks_a_co_configured_generic_client(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_CLIENT_ID", "ms-cid")
        monkeypatch.setenv("GENERIC_CLIENT_ID", "generic-cid")
        assert active_sso_provider() is ActiveSSOProvider.microsoft

    def test_generic_outranks_saml(self, monkeypatch):
        monkeypatch.setenv("GENERIC_CLIENT_ID", "generic-cid")
        monkeypatch.setenv("SAML_IDP_METADATA_URL", "https://idp.example.com/metadata")
        assert active_sso_provider() is ActiveSSOProvider.generic


class TestIdJagAssertionCaptureGap:
    @pytest.mark.parametrize("env_var", ["GOOGLE_CLIENT_ID", "MICROSOFT_CLIENT_ID", "GENERIC_CLIENT_ID"])
    def test_every_oidc_login_captures_so_has_no_gap(self, monkeypatch, env_var):
        """Google, Microsoft and generic OIDC all read the id_token off the token response, so a
        deployment on any of them must not be told to switch providers."""
        monkeypatch.setenv(env_var, "configured")
        assert id_jag_assertion_capture_gap() is None

    def test_saml_has_no_gap(self, monkeypatch: pytest.MonkeyPatch):
        """The SAML callback retains the signed assertion as the saml2 subject token, so a SAML
        deployment is a capturing one and must not be told to switch to an OIDC provider."""
        monkeypatch.setenv("SAML_IDP_METADATA_URL", "https://idp.example.com/metadata")
        assert id_jag_assertion_capture_gap() is None

    def test_no_sso_configured_reports_a_gap(self):
        gap = id_jag_assertion_capture_gap()
        assert gap is not None
        assert "no SSO provider is configured" in gap

    def test_saml_beside_an_oidc_client_has_no_gap(self, monkeypatch):
        """The callback dispatches to the OIDC client before SAML, so that is the login that runs
        and it captures; reporting SAML's gap would tell the operator to fix a path nobody takes."""
        monkeypatch.setenv("MICROSOFT_CLIENT_ID", "ms-cid")
        monkeypatch.setenv("SAML_IDP_METADATA_URL", "https://idp.example.com/metadata")
        assert id_jag_assertion_capture_gap() is None


class TestIdJagAssertionCaptureGapAtStartup:
    def test_no_provider_at_startup_is_not_yet_a_gap(self):
        assert id_jag_assertion_capture_gap_at_startup() is None

    def test_saml_at_startup_has_no_gap(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SAML_IDP_METADATA_URL", "https://idp.example.com/metadata")
        assert id_jag_assertion_capture_gap_at_startup() is None

    @pytest.mark.parametrize("env_var", ["GOOGLE_CLIENT_ID", "MICROSOFT_CLIENT_ID", "GENERIC_CLIENT_ID"])
    def test_an_oidc_provider_at_startup_has_no_gap(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "configured")
        assert id_jag_assertion_capture_gap_at_startup() is None
