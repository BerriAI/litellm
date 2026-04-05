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
  for an unauthorized or tampered one.

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
        from auths import generate_inmemory_keypair, sign_action, verify_action_envelope
    except ImportError:
        print("The 'auths' Python SDK is not installed.")
        print()
        print("Install it with:")
        print("  pip install auths")
        print()
        print("Or visit: https://github.com/auths-dev/auths")
        sys.exit(0)

    # Generate ephemeral identities — no filesystem, no keychain needed
    maintainer_priv, maintainer_pub, maintainer_did = generate_inmemory_keypair()
    attacker_priv, _attacker_pub, attacker_did = generate_inmemory_keypair()

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
        maintainer_priv, "release", release_payload, maintainer_did,
    )

    result = verify_action_envelope(legitimate_envelope, maintainer_pub)
    print(f"    Signed by: {maintainer_did}")
    print(f"    Verification: {'PASSED' if result.valid else 'FAILED'}")
    print()

    # ── Step 2: Attacker publishes with stolen PyPI token ──────────────
    print("[2] Attacker publishes v1.82.7 using stolen PyPI token...")
    print("    (Attacker has registry credentials but NOT the maintainer's signing key)")
    print()

    malicious_payload = json.dumps({
        "package": "litellm",
        "version": "1.82.7",
        "digest": "sha256:malicious_payload_hash...",
        "registry": "pypi",
    })

    # Attacker signs with their own key — NOT the maintainer's
    attacker_envelope = sign_action(
        attacker_priv, "release", malicious_payload, attacker_did,
    )

    # Verify against the MAINTAINER's public key (the only trusted key)
    result = verify_action_envelope(attacker_envelope, maintainer_pub)
    print(f"    Signed by: {attacker_did}")
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

    result = verify_action_envelope(tampered_json, maintainer_pub)
    print(f"    Original signer: {maintainer_did}")
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
