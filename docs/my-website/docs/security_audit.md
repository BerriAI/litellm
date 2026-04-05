---
title: Supply Chain Security Audit
---

# Supply Chain Security Audit

In light of the March 2026 supply chain incident, LiteLLM is committed to providing tools and guidance to help developers secure their environments.

## Local Environment Audit

If you suspect your local or server environment may have been exposed to compromised versions of LiteLLM (v1.82.7 or v1.82.8), you can use the following community-contributed tools to audit your filesystem.

### Agent Safe Check (Fingerprint Scanner)

`agent-safe-check` is a lightweight Python script designed to scan your `site-packages` for malicious `.pth` files and known attack fingerpints associated with the incident.

**Key Features:**
- **Fingerprint Matching**: Compares files against known malicious SHA256 hashes.
- **Malicious .pth Detection**: Identifies hidden execution triggers like `litellm_init.pth`.

**Usage:**

```bash
# Clone and run the auditor
git clone https://github.com/sayanget/lite-guard.git
python3 lite-guard/scan.py
```

**What it checks:**
1. Presence of `litellm_init.pth` in any directory in your `sys.path`.
2. File hash matching for the reported malicious payload (`ceNa7wMJnNHy1kRnNCcwJaFjWX3pORLfMh7xGL8TUjg`).

---

## CI/CD Security Best Practices

To prevent future supply chain attacks, we recommend the following configurations for your pipelines:

### 1. Pin Dependencies
Always use SHA hashes or specific versions for GitHub Actions and pinned versions in `requirements.txt`.

### 2. Use Trusted Publishers (OIDC)
If you are publishing your own forks or internal packages, migrate to [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/) to eliminate the need for long-lived API tokens.

### 3. Network Egress Filtering
Restrict your AI Gateway's outbound traffic to known providers. The March 2026 attack attempted to exfiltrate data to a non-official domain `models.litellm.cloud`. Monitoring for unexpected outbound DNS/HTTP requests is a strong secondary defense.

---

## Official Security Resources
- [March 2026 Incident Report](/blog/security-update-march-2026)
- [Security Policy](https://github.com/BerriAI/litellm/security/policy)
