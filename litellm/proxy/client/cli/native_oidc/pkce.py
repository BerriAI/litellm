"""PKCE (RFC 7636) and OAuth state generation.

`plain` is never generated and never accepted as a downgrade -- only S256.
"""

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Final

# RFC 7636 section 4.1.
CODE_VERIFIER_MIN_LENGTH: Final = 43
CODE_VERIFIER_MAX_LENGTH: Final = 128

# `token_urlsafe` emits base64url characters (A-Za-z0-9-_), all of which are in
# the RFC 7636 unreserved set. 64 bytes yields an ~86 character verifier, well
# inside the permitted range and far above the 256 bits of entropy floor.
_CODE_VERIFIER_ENTROPY_BYTES: Final = 64
_STATE_ENTROPY_BYTES: Final = 32

CODE_CHALLENGE_METHOD: Final = "S256"


def generate_code_verifier() -> str:
    """Generate a cryptographically random RFC 7636 code verifier."""
    return secrets.token_urlsafe(_CODE_VERIFIER_ENTROPY_BYTES)


def generate_state() -> str:
    """Generate a cryptographically random OAuth state value."""
    return secrets.token_urlsafe(_STATE_ENTROPY_BYTES)


def compute_code_challenge(code_verifier: str) -> str:
    """base64url_without_padding(SHA256(code_verifier))."""
    digest: Final = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def states_match(expected: str, received: str) -> bool:
    """Constant-time state comparison."""
    return hmac.compare_digest(expected, received)


@dataclass(frozen=True)
class PkceChallenge:
    """A generated PKCE pair plus the state bound to the same authorization request.

    Never logged, never persisted, never rendered into the browser response.
    """

    code_verifier: str
    code_challenge: str
    state: str
    code_challenge_method: str = CODE_CHALLENGE_METHOD


def generate_pkce_challenge() -> PkceChallenge:
    verifier: Final = generate_code_verifier()
    return PkceChallenge(
        code_verifier=verifier,
        code_challenge=compute_code_challenge(verifier),
        state=generate_state(),
    )
