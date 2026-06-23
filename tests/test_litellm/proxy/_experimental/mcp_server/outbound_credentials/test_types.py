"""Construction-time tests for the outbound_credentials vocabulary.

The point of the typed seam is that illegal mode/field combinations are unrepresentable:
a config missing a required field, an unknown mode, or a mismatched discriminated-union
source must fail at construction, not at resolve time. These tests pin that, plus the
CredError tag/summary surface and the derived auth_spec_kind. Each assertion fails if the
corresponding guarantee is mutated away.
"""

import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    Ambient,
    ApiKeyConfig,
    AuthConfig,
    AuthSpecKind,
    AwsSigV4Config,
    Byok,
    CredError,
    Error,
    NoneConfig,
    Ok,
    ServerSpec,
    SharedKey,
    StaticKeys,
    parse_auth_spec_kind,
)

_AUTH_CONFIG = TypeAdapter(AuthConfig)


def test_parse_auth_spec_kind_accepts_known_mode():
    result = parse_auth_spec_kind("token_exchange")
    assert isinstance(result, Ok)
    assert result.ok is AuthSpecKind.token_exchange


def test_parse_auth_spec_kind_rejects_unknown_mode():
    result = parse_auth_spec_kind("totally_made_up")
    assert isinstance(result, Error)
    assert result.error.tag == "unsupported_mode"
    assert "totally_made_up" in result.error.summary


@pytest.mark.parametrize(
    "factory, expected_tag",
    [
        (CredError.of_unauthorized, "unauthorized"),
        (CredError.of_misconfigured, "misconfigured"),
        (CredError.of_upstream_unavailable, "upstream_unavailable"),
        (CredError.of_unsupported_mode, "unsupported_mode"),
        (CredError.of_precondition_required, "precondition_required"),
        (CredError.of_not_implemented, "not_implemented"),
    ],
)
def test_crederror_factory_sets_the_matching_tag(factory, expected_tag):
    err = factory("detail text")
    assert err.tag == expected_tag
    assert "detail text" in err.summary


def test_apikeyconfig_requires_a_key_source():
    with pytest.raises(ValidationError):
        ApiKeyConfig()  # type: ignore[call-arg]


def test_sharedkey_requires_a_value():
    with pytest.raises(ValidationError):
        SharedKey()  # type: ignore[call-arg]


def test_static_keys_require_id_and_secret():
    with pytest.raises(ValidationError):
        StaticKeys(access_key_id="AKIA")  # type: ignore[call-arg]


def test_aws_sigv4_requires_a_region():
    with pytest.raises(ValidationError):
        AwsSigV4Config()  # type: ignore[call-arg]


def test_aws_sigv4_defaults_to_the_ambient_credential_chain():
    cfg = AwsSigV4Config(region="us-east-1")
    assert isinstance(cfg.credentials, Ambient)
    assert cfg.service == "bedrock-agentcore"


def test_authconfig_discriminates_on_kind():
    api_key = _AUTH_CONFIG.validate_python(
        {"kind": "api_key", "key_source": {"source": "shared", "value": "k"}}
    )
    assert isinstance(api_key, ApiKeyConfig)
    assert isinstance(api_key.key_source, SharedKey)

    none = _AUTH_CONFIG.validate_python({"kind": "none"})
    assert isinstance(none, NoneConfig)


def test_authconfig_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        _AUTH_CONFIG.validate_python({"kind": "not_a_mode"})


def test_apikeysource_discriminates_and_rejects_unknown_source():
    byok = ApiKeyConfig.model_validate({"key_source": {"source": "byok"}})
    assert isinstance(byok.key_source, Byok)

    with pytest.raises(ValidationError):
        ApiKeyConfig.model_validate({"key_source": {"source": "mystery"}})


def test_server_spec_derives_auth_spec_kind_from_config():
    spec = ServerSpec(
        server_id="s1",
        resource="https://api.example.com",
        config=NoneConfig(),
    )
    assert spec.auth_spec_kind is AuthSpecKind.none

    api_spec = ServerSpec(
        server_id="s2",
        resource="https://api.example.com",
        config=ApiKeyConfig(key_source=SharedKey(value=SecretStr("k"))),
    )
    assert api_spec.auth_spec_kind is AuthSpecKind.api_key


def test_api_key_header_placement():
    default = ApiKeyConfig(key_source=SharedKey(value=SecretStr("tok")))
    assert default.header("tok") == ("Authorization", "Bearer tok")

    raw = ApiKeyConfig(
        header_name="X-API-Key",
        value_prefix="",
        key_source=SharedKey(value=SecretStr("tok")),
    )
    assert raw.header("tok") == ("X-API-Key", "tok")


def test_configs_are_frozen():
    cfg = NoneConfig()
    with pytest.raises(ValidationError):
        cfg.kind = AuthSpecKind.api_key  # type: ignore[misc]


def test_secrets_do_not_leak_in_repr():
    key = SharedKey(value=SecretStr("super-secret"))
    assert "super-secret" not in repr(key)
    assert key.value.get_secret_value() == "super-secret"
