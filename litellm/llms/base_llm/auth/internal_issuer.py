"""Mints a self-issued workload assertion for Anthropic's ``internal_issuer`` identity source:
LiteLLM signs its own short-lived ES256 JWT instead of reading one from a mounted OIDC file.

Signing custody is the operator-supplied PEM at ``InternalIssuerSource.signing_key_ref``,
resolved the same way every other WIF secret pointer already is (env, a Credential, or
whatever secret manager ``litellm.secret_manager_client`` is globally configured to, Vault
included) -- see Phase 1 decision 1. Every mint is fresh; nothing here caches a minted JWT,
since the outer token-exchange engine already caches the Anthropic token it buys with one.
"""

import time
import uuid
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.llms.base_llm.auth.identity_source import InternalIssuerSource, ref_for_error_message
from litellm.llms.base_llm.auth.jwt_signing import jwks_document_json, sign_es256_jwt

SigningKeyReader: TypeAlias = Callable[[str], str | None]  # mutable-ok: Callable param-list syntax, not a list


def _default_signing_key_reader(ref: str) -> str | None:
    from litellm.secret_managers.main import get_secret_str

    return get_secret_str(ref)


def _claims(config: InternalIssuerSource, issued_at: int) -> Mapping[str, object]:
    return MappingProxyType(
        {
            key: value
            for key, value in (
                ("sub", config.subject),
                ("iss", config.issuer_url),
                ("aud", config.audience),
                ("iat", issued_at),
                ("exp", issued_at + config.ttl_seconds),
                ("jti", str(uuid.uuid4())),
            )
            if value is not None
        }
    )


def _resolve_signing_key(config: InternalIssuerSource, key_reader: SigningKeyReader) -> str:
    pem: Final = key_reader(config.signing_key_ref)
    if not pem:
        raise ValueError(
            f"internal_issuer signing key {ref_for_error_message(config.signing_key_ref)} could not be read"
        )
    return pem


def mint_internal_issuer_assertion(
    config: InternalIssuerSource,
    *,
    key_reader: SigningKeyReader = _default_signing_key_reader,
    clock: Callable[[], float] = time.time,
) -> str:
    """Signs one fresh, short-lived assertion; the caller must not cache the result, since a
    cached copy would defeat the point of re-minting on every exchange."""
    pem: Final = _resolve_signing_key(config, key_reader)
    return sign_es256_jwt(pem, _claims(config, issued_at=int(clock())))


def internal_issuer_assertion_source(
    config: InternalIssuerSource,
    *,
    key_reader: SigningKeyReader = _default_signing_key_reader,
    clock: Callable[[], float] = time.time,
) -> Callable[[], str]:
    """A zero-arg closure that mints fresh on every call: the shape an ``oidc/internal_issuer/...``
    ref dispatches to once wired into ``TokenExchangeSpec.assertion_source`` (Phase 1 decision 7)
    -- the caller parses the config and closes this function over it, with no registry involved."""
    return lambda: mint_internal_issuer_assertion(config, key_reader=key_reader, clock=clock)


def internal_issuer_jwks_document(
    config: InternalIssuerSource,
    *,
    key_reader: SigningKeyReader = _default_signing_key_reader,
) -> str:
    """The operator-facing JWKS export, resolved from a configured identity source rather than
    a raw PEM in hand -- the JSON document to register as Anthropic's inline federation issuer."""
    return jwks_document_json(_resolve_signing_key(config, key_reader))
