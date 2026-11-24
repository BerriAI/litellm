# Data Privacy and Security

## Security Announcements

LiteLLM maintains a security announcements mailing list that is open to anyone. Subscribers receive advance notice, typically one to two days, before we release a fix for a particularly severe vulnerability or for any vulnerability exploitable by an unauthenticated attacker. This notice is provided on a best-effort basis

To subscribe, visit [https://security.litellm.ai/mailinglist/](https://security.litellm.ai/mailinglist/)

## Security Vulnerability Reporting Guidelines


We value the security community's role in protecting our systems and users.
To report a security vulnerability, please submit a report through our vulnerability disclosure form:

[LiteLLM VDP](https://security.litellm.ai/vdp/)

These reports are submitted to HackerOne and are triaged there, though no account is needed.

**Please do not submit vulnerabilities as draft GHSAs. As of 2026-11-11 new GHSA submissions will be closed.**

### Vulnerability Categories

We classify vulnerabilities into the following categories:

**P0: Supply Chain Attacks**

Attacks that compromise our CI/CD pipeline, allowing a malicious actor to point our PyPI package or Docker images (GHCR or Docker Hub) to vulnerable or tampered artifacts.

**P1: Unauthenticated Proxy Access**

Application-level attacks where an unauthenticated user is able to gain access to protected data on a LiteLLM proxy instance that should be protected (e.g api keys).

**P2: Authenticated Malicious Actions**

Application-level attacks where an authenticated user is able to perform actions beyond their intended permissions, such as privilege escalation or unauthorized data access.

### Bug Bounty Program

We offer bounties for responsibly disclosed vulnerabilities based on severity:

**Note that currently only P0/P1 reports are eligible for a bounty, though submissions for P2 bugs are still encouraged**

| Severity | Bounty Range | Example |
|----------|-------------|---------|
| **Critical** | $1,500 - $3,000 | P0 supply chain compromise |
| **High** | $500 - $1,500 | P1 unauthenticated proxy access |
| **Medium** | N/A | P2 authenticated privilege escalation |
| **Low** | N/A | Minor information disclosure, low-impact misconfigurations |

To qualify for a bounty, reports must include clear reproduction steps, a reproduction video as described above, and must not involve systems or accounts you do not own. We review all submissions promptly and will follow up within 5 business days.

### Known Non-Issues

- **Misconfigurations:** attacks that require misconfiguration of the application (e.g. not setting a `master_key` on the proxy configuration).
- **Insecure configurations:** attacks that require deliberate configuration changes to disable security functionality. (e.g. IDOR on response IDs requiring configuration of `disable_responses_id_security` to `true`)
- **Social engineering:** attacks that rely on social engineering as their primary mechanism (e.g. convincing a user's agent to install a malicious package we did not publish).

## Security Measures

### LiteLLM Github

- All commits run through Github's CodeQL checking

### Release Signing

Every release published since November 15, 2026, starting with `v1.111.0-dev.3`, is signed with keyless [cosign](https://docs.sigstore.dev/cosign/overview/). This includes patch releases on older lines and the Helm chart at `ghcr.io/berriai/litellm/chart/litellm`. There is no signing key. Each release run gets a short-lived certificate from Sigstore bound to our release workflow, and the signature is recorded in the public Rekor transparency log. Verify a release with:

```bash
cosign verify \
  --certificate-identity https://github.com/BerriAI/project-releaser/.github/workflows/release.yml@refs/heads/main \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/berriai/litellm:<release-tag>
```

`cosign verify --key` does not work on these releases.

Releases published before November 15, 2026 keep their key signature. Verify them with the `cosign.pub` from the commit that introduced the key. Releases published from October 24, 2026 on use the key from commit [`e82a3c6`](https://github.com/BerriAI/litellm/commit/e82a3c646908ba0a6ee0beddc4310d7a6cac4065), whose `cosign.pub` has the SHA-256 fingerprint `e3fe3e08c44b16407fe0b37c43b4f8b603b3a0f4ee74c29e647bc022d762c2c0`:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/e82a3c646908ba0a6ee0beddc4310d7a6cac4065/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

Earlier releases use the previous key from commit [`0112e53`](https://github.com/BerriAI/litellm/commit/0112e53046018d726492c814b3644b7d376029d0), at `https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub`. Trust a signature from the previous key only on an image published before `v1.107.0` on October 21, 2026, as explained in the [incident post](https://docs.litellm.ai/blog/security-incident-oct-21-2026). [Discussion #48242](https://github.com/BerriAI/litellm/discussions/48242) lists which releases are signed with which key.

### Self-hosted Instances LiteLLM

- **No data or telemetry is stored on LiteLLM Servers when you self host**
- For installation and configuration, see: [Self-hosting guided](https://docs.litellm.ai/docs/proxy/deploy)
- **Telemetry** We run no telemetry when you self host LiteLLM

### LiteLLM Cloud

- We encrypt all data stored using your `LITELLM_MASTER_KEY` and in transit using TLS.
- Our database and application run on GCP, AWS infrastructure, partly managed by NeonDB.
    - US data region: Northern California (AWS/GCP `us-west-1`) & Virginia (AWS `us-east-1`)
    - EU data region Germany/Frankfurt (AWS/GCP `eu-central-1`)
- All users have access to SSO (Single Sign-On) through OAuth 2.0 with Google, Okta, Microsoft, KeyCloak. 
- Audit Logs with retention policy
- Control Allowed IP Addresses that can access your Cloud LiteLLM Instance

For security inquiries, please contact us at security@berri.ai

#### Supported data regions for LiteLLM Cloud

LiteLLM supports the following data regions:

- US, Northern California (AWS/GCP `us-west-1`)
- Europe, Frankfurt, Germany (AWS/GCP `eu-central-1`)

All data, user accounts, and infrastructure are completely separated between these two regions
