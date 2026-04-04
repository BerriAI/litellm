# Security Cookbook: Auths Commit Verification

## Background

On March 24, 2026, LiteLLM was the target of a supply chain attack. The attacker
compromised the Trivy GitHub Action, which exfiltrated the `PYPI_PUBLISH` token from
LiteLLM's CI/CD pipeline. The stolen token was used to publish malicious versions
(v1.82.7 and v1.82.8) directly to PyPI. The source code on GitHub was never modified.

The attack succeeded because **there was no cryptographic binding between the published
package and a verified maintainer identity**.

## What is Auths?

[Auths](https://github.com/auths-dev/auths) provides Ed25519 signatures bound to
KERI-based decentralized identifiers (DIDs). With Auths:

- Every commit carries a signature from the maintainer's cryptographic identity
- The signature is bound to the maintainer's device keychain (not a registry account)
- Stealing PyPI/npm credentials is insufficient without the signing key
- Verification happens locally — no network calls to a central authority

## Running the Simulation

The simulation script recreates the attack scenario and demonstrates how Auths
verification catches the unauthorized commit:

```bash
brew tap auths-dev/auths-cli && brew install auths
python auths_attack_simulation.py
```

## Adding Auths to Your Workflow

See the GitHub Actions workflow at `.github/workflows/auths-verify-commits.yml`
and the allowed signers configuration at `.auths/allowed_signers`.
