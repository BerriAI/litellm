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
    ClientSecretAuth,
    CredError,
    Error,
    IdJagConfig,
    NoneConfig,
    Ok,
    PrivateKeyJwtAuth,
    ServerSpec,
    SharedKey,
    StaticKeys,
    parse_auth_spec_kind,
)

_AUTH_CONFIG = TypeAdapter(AuthConfig)

_ID_JAG_MINIMAL = {
    "kind": "id_jag",
    "org_token_endpoint": "https://idp.example.com/token",
    "resource_token_endpoint": "https://mcp-as.example.com/token",
    "client_id": "litellm",
    "client_auth": {"source": "client_secret", "client_secret": "s"},
}


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


@pytest.mark.parametrize(
    "missing",
    ["org_token_endpoint", "resource_token_endpoint", "client_id", "client_auth"],
)
def test_id_jag_config_requires_each_endpoint_client_and_auth(missing):
    payload = {k: v for k, v in _ID_JAG_MINIMAL.items() if k != missing}
    with pytest.raises(ValidationError):
        _AUTH_CONFIG.validate_python(payload)


def test_id_jag_client_auth_discriminates_on_source():
    by_secret = _AUTH_CONFIG.validate_python(_ID_JAG_MINIMAL)
    assert isinstance(by_secret, IdJagConfig)
    assert isinstance(by_secret.client_auth, ClientSecretAuth)
    assert by_secret.client_auth.client_secret.get_secret_value() == "s"

    by_key = _AUTH_CONFIG.validate_python(
        {
            **_ID_JAG_MINIMAL,
            "client_auth": {
                "source": "private_key_jwt",
                "private_key": "PEM",
                "key_id": "kid-1",
                "signing_alg": "RS384",
            },
        }
    )
    assert isinstance(by_key, IdJagConfig)
    assert isinstance(by_key.client_auth, PrivateKeyJwtAuth)
    assert by_key.client_auth.private_key.get_secret_value() == "PEM"
    assert by_key.client_auth.key_id == "kid-1"
    assert by_key.client_auth.signing_alg == "RS384"


def test_id_jag_client_auth_rejects_unknown_source():
    with pytest.raises(ValidationError):
        _AUTH_CONFIG.validate_python(
            {**_ID_JAG_MINIMAL, "client_auth": {"source": "mystery"}}
        )


def test_id_jag_config_defaults_id_token_subject_and_empty_optionals():
    config = _AUTH_CONFIG.validate_python(_ID_JAG_MINIMAL)
    assert isinstance(config, IdJagConfig)
    assert config.subject_token_type == "urn:ietf:params:oauth:token-type:id_token"
    assert config.audience is None
    assert config.resource is None
    assert config.scopes == ()


def test_id_jag_secrets_do_not_leak_in_repr():
    config = IdJagConfig(
        org_token_endpoint="https://idp.example.com/token",
        resource_token_endpoint="https://mcp-as.example.com/token",
        client_id="litellm",
        client_auth=PrivateKeyJwtAuth(private_key=SecretStr("super-secret-pem")),
    )
    assert "super-secret-pem" not in repr(config)


def test_id_jag_server_spec_derives_auth_spec_kind():
    config = _AUTH_CONFIG.validate_python(_ID_JAG_MINIMAL)
    spec = ServerSpec(
        server_id="s",
        resource="https://mcp.example.com/mcp",
        config=config,
    )
    assert spec.auth_spec_kind is AuthSpecKind.id_jag
