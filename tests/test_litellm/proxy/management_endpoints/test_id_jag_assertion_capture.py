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
        """The callback tests GOOGLE_CLIENT_ID first, so the generic arm never runs here and
        no assertion is captured; reporting generic would clear a gap that is still open."""
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
    def test_generic_oidc_has_no_gap(self, monkeypatch):
        monkeypatch.setenv("GENERIC_CLIENT_ID", "generic-cid")
        assert id_jag_assertion_capture_gap() is None

    @pytest.mark.parametrize(
        "env_var, provider_label",
        [
            ("GOOGLE_CLIENT_ID", "google"),
            ("MICROSOFT_CLIENT_ID", "microsoft"),
            ("SAML_IDP_METADATA_URL", "saml"),
        ],
    )
    def test_non_capturing_provider_is_named_with_the_remedy(self, monkeypatch, env_var, provider_label):
        monkeypatch.setenv(env_var, "configured")
        gap = id_jag_assertion_capture_gap()
        assert gap is not None
        assert provider_label in gap
        assert "GENERIC_CLIENT_ID" in gap

    def test_no_sso_configured_reports_a_gap(self):
        gap = id_jag_assertion_capture_gap()
        assert gap is not None
        assert "no SSO provider is configured" in gap

    def test_google_beside_generic_still_reports_a_gap(self, monkeypatch):
        """The precedence trap in operator terms: adding a generic client id without removing
        GOOGLE_CLIENT_ID does not fix the deployment, so the gap must not clear."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-cid")
        monkeypatch.setenv("GENERIC_CLIENT_ID", "generic-cid")
        gap = id_jag_assertion_capture_gap()
        assert gap is not None
        assert "google" in gap


class TestIdJagAssertionCaptureGapAtStartup:
    def test_no_provider_at_startup_is_not_yet_a_gap(self):
        assert id_jag_assertion_capture_gap_at_startup() is None

    def test_google_provider_at_startup_reports_the_capture_gap(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-cid")
        startup_gap = id_jag_assertion_capture_gap_at_startup()
        callback_gap = id_jag_assertion_capture_gap()
        assert startup_gap is not None
        assert startup_gap == callback_gap

    def test_generic_provider_at_startup_has_no_gap(self, monkeypatch):
        monkeypatch.setenv("GENERIC_CLIENT_ID", "generic-cid")
        assert id_jag_assertion_capture_gap_at_startup() is None
