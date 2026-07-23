"""Provenance of the inbound subject token, decided at the resolver's v1 edge.

The resolver must never forward a gateway-issued or gateway-honored credential to an external
token endpoint as exchange subject material. Recognizing "is this a credential this gateway
issued" by an enumerated denylist of prefixes is a leak generator: the set of minted formats
grows, and every format the list has not caught yet is disclosed. So this module recognizes a
gateway credential by the one property every one of them shares and no external token shares:
it is cryptographically ours, signed or encrypted under this gateway's master key. A new mint
format is caught by construction, with no edit here.

``classify_inbound_provenance`` is the single owner both exchange arms read through
(``Subject.inbound_provenance``). id_jag's subject is an id_token (a JWT), so it exchanges the
inbound bearer only when it is ``external_jwt`` and otherwise sources the stored assertion (an
``external_opaque`` value is not an id_token and is never forwarded); token_exchange (OBO) keeps
its exchange-what-was-presented contract for both ``external_jwt`` and ``external_opaque`` tokens
but refuses a ``gateway_credential``.
"""

from __future__ import annotations

import hashlib
import secrets

import jwt

from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    ENVELOPE_ISSUER,
    ENVELOPE_PREFIX,
    REFRESH_ENVELOPE_PREFIX,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    SESSION_ISSUER,
    SESSION_REFRESH_PREFIX,
    SESSION_TOKEN_PREFIX,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import InboundTokenProvenance

_VIRTUAL_KEY_PREFIX = "sk-"
_GATEWAY_MINT_PREFIXES = (
    _VIRTUAL_KEY_PREFIX,
    SESSION_TOKEN_PREFIX,
    SESSION_REFRESH_PREFIX,
    ENVELOPE_PREFIX,
    REFRESH_ENVELOPE_PREFIX,
)
_RESERVED_GATEWAY_ISSUERS = frozenset({SESSION_ISSUER, ENVELOPE_ISSUER})

# Recognition asks whether a token is THIS gateway's, not whether it is still usable, so every
# decode below disables claim validation (expiry, not-before, issued-at, audience, issuer). An
# expired or not-yet-valid gateway credential is still a gateway credential; letting a claim check
# reject it would drop it back to the exchangeable-external path and disclose it upstream. Only the
# signature (for the master-key check) speaks to provenance.
_RECOGNITION_CLAIM_CHECKS_DISABLED = {
    "verify_exp": False,
    "verify_nbf": False,
    "verify_iat": False,
    "verify_aud": False,
    "verify_iss": False,
}


def _master_key() -> str | None:
    from litellm.proxy.proxy_server import master_key

    return master_key


def _equals_master_key(token: str, master_key: str) -> bool:
    """Constant-time equality with the master key, TOTAL over any input string.

    ``secrets.compare_digest`` raises ``TypeError`` on a ``str`` carrying non-ASCII characters, and
    recognition must never raise into egress (an inbound bearer is fully attacker-controlled). Both
    sides are hashed to fixed-length digests first, so any string compares without raising and the
    master key's length is not leaked; a non-ASCII bearer reads "not the master key" rather than
    500-ing the request.
    """
    token_digest = hashlib.sha256(token.encode("utf-8", "surrogatepass")).digest()
    master_key_digest = hashlib.sha256(master_key.encode("utf-8", "surrogatepass")).digest()
    return secrets.compare_digest(token_digest, master_key_digest)


def _verifies_under_master_key(token: str, master_key: str) -> bool:
    """True when ``token`` is a JWT this gateway signed with the master key (HS256).

    Covers every master-key-signed mint (UI login session, onboarding, byok_session) and any
    future one, since the check is the SIGNATURE, not the format and not the claims. Claim
    validation is disabled: an expired or not-yet-valid gateway token is still this gateway's and
    must be recognized, or it would fall through to the exchangeable-external path and leak.
    """
    try:
        jwt.decode(
            token,
            master_key,
            algorithms=["HS256"],
            options={"verify_signature": True, **_RECOGNITION_CLAIM_CHECKS_DISABLED},
        )
        return True
    except Exception:  # noqa: BLE001  # signature (not claims) failed -> not signed by us; other checks still run
        return False


def _claims_a_reserved_issuer(token: str) -> bool:
    """True when ``token`` is a JWT whose ``iss`` is one this gateway reserves (session/envelope).

    These are signed with master-key-DERIVED keys, so they do not verify under the master key
    directly; they are already caught by their mint prefix, and this is the belt-and-suspenders
    for an inner token presented without its prefix. The issuer is refused on the unverified
    claim alone because the value is reserved to this gateway: no external token legitimately
    carries it, so treating it as ours fails closed. Claim validation stays off for the same
    reason as the master-key check: an expired reserved-issuer token is still ours.
    """
    try:
        claims = jwt.decode(token, options={"verify_signature": False, **_RECOGNITION_CLAIM_CHECKS_DISABLED})
    except Exception:  # noqa: BLE001  # not a JWT
        return False
    return claims.get("iss") in _RESERVED_GATEWAY_ISSUERS


def _decrypts_under_gateway_key(token: str) -> bool:
    """True when ``token`` decrypts under this gateway's salt key (the CLI / experimental UI
    login blobs, which are encrypted rather than JWT-signed). Decryption succeeding is proof the
    value is ours; an external token decrypts to nothing."""
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    try:
        decrypted = decrypt_value_helper(
            token, key="mcp_subject_token_provenance", exception_type="debug", return_original_value=False
        )
    except Exception:  # noqa: BLE001  # defensive: never let recognition raise into egress
        return False
    return decrypted is not None


def is_gateway_issued_credential(token: str) -> bool:
    """Whether ``token`` is a credential this gateway issued or honors as its own.

    Complete by construction rather than by enumeration: a gateway credential is caught if it is
    the master key, carries a gateway mint prefix, verifies under the master key, claims a
    reserved gateway issuer, or decrypts under the gateway salt key. Every current mint (virtual
    key, master key, MCP session/refresh, bridge envelope, UI login session, onboarding,
    byok_session, CLI/experimental login) matches at least one, and a future mint signed or
    encrypted with the master key matches without a change here.
    """
    master_key = _master_key()
    if master_key and _equals_master_key(token, master_key):
        return True
    if token.startswith(_GATEWAY_MINT_PREFIXES):
        return True
    if master_key and _verifies_under_master_key(token, master_key):
        return True
    if _claims_a_reserved_issuer(token):
        return True
    return _decrypts_under_gateway_key(token)


def _is_decodable_jwt(token: str) -> bool:
    """Whether ``token`` is structurally a JWT (an id_token is a JWT). Signature and every
    registered-claim check are disabled: this is a shape test to tell an id_token candidate from
    an opaque bearer, not a validation. The gateway check runs first, so this only ever runs on a
    non-gateway token; the org authorization server remains the authority on the id_token itself.
    """
    try:
        jwt.decode(token, options={"verify_signature": False, **_RECOGNITION_CLAIM_CHECKS_DISABLED})
        return True
    except Exception:  # noqa: BLE001  # not a JWT -> an opaque bearer, not an id_token candidate
        return False


def classify_inbound_provenance(subject_token: str | None) -> InboundTokenProvenance:
    """The one classification both exchange arms read. A gateway-issued credential is never
    exchange subject material (any mode). Among the caller's own credentials, a JWT is an id_token
    candidate (``external_jwt``) that id_jag can exchange, while an opaque value (``external_opaque``)
    is not an id_token and id_jag falls to the stored assertion; OBO may forward either."""
    if not subject_token:
        return "absent"
    if is_gateway_issued_credential(subject_token):
        return "gateway_credential"
    if _is_decodable_jwt(subject_token):
        return "external_jwt"
    return "external_opaque"
