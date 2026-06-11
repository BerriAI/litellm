from __future__ import annotations

import pytest
from pydantic import ValidationError

from litellm.proxy.auth_v2.config import (
    OAuth2IntrospectionConfig,
)
from litellm.proxy.auth_v2 import OIDCProviderConfig, SAMLConfig


def test_saml_config_requires_idp_metadata_when_enabled():
    with pytest.raises(ValidationError):
        SAMLConfig(enabled=True, entity_id="sp", acs_url="https://sp/acs")


def test_saml_config_allows_empty_metadata_when_disabled():
    config = SAMLConfig(enabled=False, entity_id="sp", acs_url="https://sp/acs")
    assert config.idp_metadata == ""


def test_saml_config_accepts_inline_metadata():
    config = SAMLConfig(
        enabled=True,
        entity_id="sp",
        acs_url="https://sp/acs",
        idp_metadata="<EntityDescriptor/>",
    )
    assert config.idp_metadata == "<EntityDescriptor/>"


def test_oidc_provider_requires_audience():
    with pytest.raises(ValidationError):
        OIDCProviderConfig(issuer="https://idp.example.com")


def test_oidc_provider_defaults_to_rs256():
    provider = OIDCProviderConfig(issuer="https://idp.example.com", audience=["x"])
    assert provider.algorithms == ["RS256"]
    assert provider.require_at_jwt is False


def test_introspection_client_secret_is_secret():
    config = OAuth2IntrospectionConfig(
        introspection_endpoint="https://idp.example.com/introspect",
        client_id="rp",
        client_secret="hunter2",
    )
    # SecretStr never leaks the value in its repr
    assert "hunter2" not in repr(config)
    assert config.client_secret.get_secret_value() == "hunter2"
