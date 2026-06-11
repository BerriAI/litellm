from __future__ import annotations

from typing import Any, Tuple

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy.auth_v2.authenticators import JWTVerifier
from litellm.proxy.auth_v2 import OIDCProviderConfig

from auth_v2_helpers import TEST_AUDIENCE, TEST_ISSUER, FakeJwksClient, TokenFactory


def _generate_keypair() -> Tuple[bytes, Any]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return private_pem, private_key.public_key()


@pytest.fixture(scope="session")
def rsa_keypair() -> Tuple[bytes, Any]:
    return _generate_keypair()


@pytest.fixture(scope="session")
def other_rsa_keypair() -> Tuple[bytes, Any]:
    return _generate_keypair()


@pytest.fixture
def token_factory(rsa_keypair: Tuple[bytes, Any]) -> TokenFactory:
    private_pem, _ = rsa_keypair
    return TokenFactory(private_pem)


@pytest.fixture
def oidc_provider() -> OIDCProviderConfig:
    return OIDCProviderConfig(issuer=TEST_ISSUER, audience=[TEST_AUDIENCE])


@pytest.fixture
def jwt_verifier(
    rsa_keypair: Tuple[bytes, Any], oidc_provider: OIDCProviderConfig
) -> JWTVerifier:
    _, public_key = rsa_keypair
    return JWTVerifier(oidc_provider, jwks_client=FakeJwksClient(public_key))
