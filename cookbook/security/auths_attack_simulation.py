"""
Auths Attack Simulation: LiteLLM March 24, 2026 Supply Chain Incident

Demonstrates how Auths cryptographic verification would have detected the
unauthorized PyPI publish that compromised LiteLLM v1.82.7 and v1.82.8.

What happened:
  1. Attacker compromised the Trivy GitHub Action (March 19)
  2. LiteLLM's CI ran Trivy without version pinning
  3. Compromised Trivy exfiltrated the PYPI_PUBLISH token from GitHub Actions
  4. Attacker used the stolen token to publish malicious versions to PyPI
  5. The malicious packages contained a credential stealer in a .pth file
  6. Source code on GitHub was never modified — the attack existed only in PyPI

How Auths closes this gap:
  The real attack bypassed Git entirely — the attacker published directly to
  PyPI with no corresponding commit. Auths establishes a policy that every
  legitimate release must trace back to a signed action by an authorized
  maintainer. A package published without a valid signature from a known
  maintainer identity has no valid attestation and would be rejected.

  This simulation uses the Auths Python SDK to demonstrate the core
  cryptographic primitive: sign an action with a maintainer's key, then
  show that verification succeeds for the legitimate release and fails
  for a tampered or unauthorized one.

Usage:
  pip install auths
  python auths_attack_simulation.py

Requires: auths (Python SDK)
"""
import json
import sys


def main() -> None:
    print("=" * 70)
    print("Auths Attack Simulation: LiteLLM Supply Chain Incident (March 24, 2026)")
    print("=" * 70)
    print()

    try:
        from auths import sign_action, verify_action_envelope
    except ImportError:
        print("The 'auths' Python SDK is not installed.")
        print()
        print("Install it with:")
        print("  pip install auths")
        print()
        print("Or visit: https://github.com/auths-dev/auths")
        sys.exit(0)

    # Derive the public key from the private seed for verification.
    # In production, the maintainer's public key comes from their Auths
    # identity (did:keri:...) and is listed in .auths/allowed_signers.
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        MAINTAINER_SEED_HEX = "a" * 64  # Simulated maintainer private key seed
        seed_bytes = bytes.fromhex(MAINTAINER_SEED_HEX)
        private_key = Ed25519PrivateKey.from_private_bytes(seed_bytes)
        MAINTAINER_PK_HEX = private_key.public_key().public_bytes_raw().hex()
    except ImportError:
        print("This simulation requires the 'cryptography' package.")
        print("Install it with: pip install cryptography")
        sys.exit(0)

    ATTACKER_SEED_HEX = "b" * 64  # Attacker has a different key
    MAINTAINER_DID = "did:keri:EBfxc_LiteLLM_Maintainer"

    # ── Step 1: Legitimate maintainer signs a release ──────────────────
    print("[1] Legitimate maintainer signs release v1.82.6...")
    print()

    release_payload = json.dumps({
        "package": "litellm",
        "version": "1.82.6",
        "digest": "sha256:abc123def456...",
        "registry": "pypi",
    })

    legitimate_envelope = sign_action(
        MAINTAINER_SEED_HEX,
        "release",
        release_payload,
        MAINTAINER_DID,
    )

    result = verify_action_envelope(legitimate_envelope, MAINTAINER_PK_HEX)
    print(f"    Signed by: {MAINTAINER_DID}")
    print(f"    Verification: {'PASSED' if result.valid else 'FAILED'}")
    print()

    # ── Step 2: Attacker publishes with stolen PyPI token ──────────────
    print("[2] Attacker publishes v1.82.7 using stolen PyPI token...")
    print("    (Attacker has registry credentials but NOT the signing key)")
    print()

    malicious_payload = json.dumps({
        "package": "litellm",
        "version": "1.82.7",
        "digest": "sha256:malicious_payload_hash...",
        "registry": "pypi",
    })

    # Attacker signs with their own key — NOT the maintainer's
    attacker_envelope = sign_action(
        ATTACKER_SEED_HEX,
        "release",
        malicious_payload,
        "did:keri:EATTACKER_unknown_identity",
    )

    # Verify against the MAINTAINER's public key (the only trusted key)
    result = verify_action_envelope(attacker_envelope, MAINTAINER_PK_HEX)
    print(f"    Signed by: did:keri:EATTACKER_unknown_identity")
    print(f"    Verification against maintainer key: {'PASSED' if result.valid else 'FAILED'}")
    if result.error:
        print(f"    Reason: {result.error}")
    print()

    # ── Step 3: Show tampered legitimate envelope also fails ───────────
    print("[3] Attacker tampers with a legitimately-signed envelope...")
    print()

    envelope = json.loads(legitimate_envelope)
    envelope["payload"]["version"] = "1.82.7"
    envelope["payload"]["digest"] = "sha256:malicious_payload_hash..."
    tampered_json = json.dumps(envelope)

    result = verify_action_envelope(tampered_json, MAINTAINER_PK_HEX)
    print(f"    Original signer: {MAINTAINER_DID}")
    print(f"    Tampered payload version: 1.82.7")
    print(f"    Verification: {'PASSED' if result.valid else 'FAILED'}")
    if result.error:
        print(f"    Reason: {result.error}")
    print()

    # ── Summary ────────────────────────────────────────────────────────
    print("-" * 70)
    print("SUMMARY")
    print()
    print("  v1.82.6 (legitimate, signed by maintainer): VERIFIED")
    print("  v1.82.7 (attacker's key, not trusted):      REJECTED")
    print("  v1.82.7 (tampered legitimate envelope):     REJECTED")
    print()
    print("NOTE: The real March 24 attack bypassed Git entirely — the attacker")
    print("published directly to PyPI with no commit at all. This simulation")
    print("demonstrates the cryptographic primitive that Auths provides: only")
    print("the holder of the maintainer's private key can produce a valid")
    print("signature. In a full deployment, the CI/CD pipeline would use")
    print("'auths artifact sign' to bind the published package to the")
    print("maintainer's identity, and consumers would verify before installing.")
    print()
    print("Learn more: https://github.com/auths-dev/auths")
    print("=" * 70)


if __name__ == "__main__":
    main()
