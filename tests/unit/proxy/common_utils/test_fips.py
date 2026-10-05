import pytest

from litellm.proxy.common_utils.fips import (
    FipsModeError,
    FipsModeOff,
    FipsModeOn,
    MalformedFipsMode,
    ProviderDoesNotEnforceFips,
    TlsVerificationDisabled,
    enforce_fips_boot_verdict,
    fips_boot_verdict,
    is_fips_mode,
    parse_fips_mode,
)


def _verdict(
    raw: str | None,
    *,
    enforcing: bool = True,
    ssl_env: str | None = None,
    ssl_setting: object = True,
):
    return fips_boot_verdict(
        raw_fips_mode=raw,
        provider_enforces_fips=lambda: enforcing,
        ssl_verify_environment=ssl_env,
        ssl_verify_setting=ssl_setting,
    )


@pytest.mark.parametrize("raw", [None, "false", "0", "no", "off", "", " False "])
def test_unset_and_false_spellings_leave_fips_mode_off(raw):
    assert parse_fips_mode(raw) == FipsModeOff()
    assert is_fips_mode({"LITELLM_FIPS_MODE": raw}.get) is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", " TRUE "])
def test_true_spellings_turn_fips_mode_on(raw):
    assert parse_fips_mode(raw) == FipsModeOn()
    assert is_fips_mode({"LITELLM_FIPS_MODE": raw}.get) is True


@pytest.mark.parametrize("raw", ["enforced", "2", "strict", "yes please"])
def test_anything_else_is_malformed_and_refused_with_the_offending_value(raw):
    assert parse_fips_mode(raw) == MalformedFipsMode(value=raw)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_boot_verdict(_verdict(raw, enforcing=False), announce=lambda _: None)
    assert f"LITELLM_FIPS_MODE={raw} is not a boolean" in str(refused.value)
    assert "true or false" in str(refused.value)


def test_off_never_consults_the_provider_or_tls_settings():
    def explode() -> bool:
        raise AssertionError("provider probe must not run while FIPS mode is off")

    verdict = fips_boot_verdict(
        raw_fips_mode=None, provider_enforces_fips=explode, ssl_verify_environment="false", ssl_verify_setting=False
    )
    assert verdict == FipsModeOff()
    enforce_fips_boot_verdict(verdict, announce=lambda _: pytest.fail("nothing to announce when off"))


def test_on_with_an_enforcing_provider_and_verified_tls_boots():
    verdict = _verdict("true", enforcing=True, ssl_env="true", ssl_setting="/etc/ssl/certs/ca.pem")
    assert verdict == FipsModeOn()
    enforce_fips_boot_verdict(verdict, announce=lambda _: pytest.fail("nothing to announce when on"))


def test_on_with_a_non_enforcing_provider_is_refused_and_names_the_fix():
    announced = []
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_boot_verdict(_verdict("true", enforcing=False), announce=announced.append)
    message = str(refused.value)
    assert message.startswith("LiteLLM proxy refused to start")
    assert "LITELLM_FIPS_MODE is on but this Python does not enforce FIPS" in message
    assert "FIPS image" in message
    assert announced == [f"\n{message}\n\n"]


@pytest.mark.parametrize(
    "ssl_env, ssl_setting, sources",
    [
        ("false", True, ("SSL_VERIFY",)),
        (" FALSE ", True, ("SSL_VERIFY",)),
        (None, False, ("litellm_settings.ssl_verify",)),
        (None, "False", ("litellm_settings.ssl_verify",)),
        ("false", False, ("SSL_VERIFY", "litellm_settings.ssl_verify")),
    ],
)
def test_disabled_tls_verification_is_refused_naming_every_source(ssl_env, ssl_setting, sources):
    verdict = _verdict("true", enforcing=True, ssl_env=ssl_env, ssl_setting=ssl_setting)
    assert verdict == TlsVerificationDisabled(sources=sources)
    with pytest.raises(FipsModeError) as refused:
        enforce_fips_boot_verdict(verdict, announce=lambda _: None)
    assert "TLS certificate verification is disabled by " + " and ".join(sources) in str(refused.value)


@pytest.mark.parametrize("ssl_setting", [True, "true", "/etc/ssl/certs/ca.pem", None, "", "0", "no"])
def test_verified_or_custom_bundle_tls_settings_are_not_treated_as_disabled(ssl_setting):
    assert _verdict("true", enforcing=True, ssl_setting=ssl_setting) == FipsModeOn()


def test_disabled_tls_is_reported_before_the_provider_so_operators_see_config_mistakes_first():
    assert _verdict("true", enforcing=False, ssl_env="false") == TlsVerificationDisabled(sources=("SSL_VERIFY",))
    assert _verdict("true", enforcing=False) == ProviderDoesNotEnforceFips()
