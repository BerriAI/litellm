# Data Privacy and Security


## Security Vulnerability Reporting Guidelines

> [!WARNING]
> Reports that do not include a video demonstrating the exploit will be closed without review. See [Reproduction Video Requirement](#reproduction-video-requirement) below.

We value the security community's role in protecting our systems and users. To report a security vulnerability:

- File a private vulnerability report on GitHub: [Report a vulnerability](https://github.com/BerriAI/litellm/security/advisories/new)
- Include steps to reproduce the issue
- Include a video or screen recording demonstrating the full exploit against a live LiteLLM instance, from initial access through to impact. A terminal recording (for example asciinema) is fine for CLI-only exploits.
- Provide any relevant additional information

### Reproduction Video Requirement

A video demonstrating the exploit is required for every report. AI tools have made it easy to produce plausible-sounding vulnerability reports that do not reproduce in practice, and triaging them takes time away from real issues. Reports submitted without a working reproduction video will be closed without review. If you add a video to a closed report, we will reopen and triage it.

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

- Attacks that require a misconfiguration on setup (e.g not setting a `master_key` on the proxy configuration), are **explicitly not in scope** and are not considered vulnerable.

## Security Measures

### LiteLLM Github

- All commits run through Github's CodeQL checking

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

For security inquiries, please contact us at support@berri.ai

#### Supported data regions for LiteLLM Cloud

LiteLLM supports the following data regions:

- US, Northern California (AWS/GCP `us-west-1`)
- Europe, Frankfurt, Germany (AWS/GCP `eu-central-1`)

All data, user accounts, and infrastructure are completely separated between these two regions
