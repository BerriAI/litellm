---
slug: security-hardening-april-2026
title: "Security Update: Vulnerability Disclosures and Ongoing Hardening"
date: 2026-04-03T12:00:00
authors:
  - krrish
  - ishaan-alt
description: "Disclosure of security vulnerabilities fixed in LiteLLM v1.83.0, and the launch of our bug bounty program."
tags: [security]
hide_table_of_contents: false
---

After the [supply chain incident](https://docs.litellm.ai/blog/security-update-march-2026) in March, we brought in [Veria Labs](https://verialabs.com/) to audit the LiteLLM proxy and fixed a number of vulnerability reports from independent researchers. All issues below are fixed in v1.83.0. If you are affected, particularly if you have JWT auth enabled, we recommend upgrading.

We've also launched a [bug bounty program](#bug-bounty-program) and Veria Labs is continuing to audit the proxy. More fixes will ship in upcoming versions.

The two high-severity issues ([CVE-2026-35029](https://github.com/BerriAI/litellm/security/advisories/GHSA-53mr-6c8q-9789) and [GHSA-69x8-hrgq-fjj8](https://github.com/BerriAI/litellm/security/advisories/GHSA-69x8-hrgq-fjj8)) **both require the attacker to already have a valid API key for the proxy**. These are not exploitable by unauthenticated users. 

The critical-severity issue ([CVE-2026-35030](https://github.com/BerriAI/litellm/security/advisories/GHSA-jjhc-v7c2-5hh6)) is an authentication bypass, but only affects deployments with `enable_jwt_auth` explicitly enabled, which is off by default. **The default LiteLLM configuration is not affected, and no LiteLLM Cloud customers had this feature enabled.**

{/* truncate */}

## Vulnerabilities

### CVE-2026-35030: Authentication bypass via OIDC cache collision (Critical)

Found by Veria Labs.

When `enable_jwt_auth` is enabled, LiteLLM cached OIDC userinfo using `token[:20]` as the cache key. JWTs from the same signing algorithm share the same header prefix, so an attacker could forge a token that hits another user's cache entry and inherit their session. We fixed this by keying the cache on `sha256(token)` instead.

**Most deployments are not affected.** This requires `enable_jwt_auth: true`, which is off by default. If you can't upgrade, disable JWT auth as a workaround.

Full advisory: [GHSA-jjhc-v7c2-5hh6](https://github.com/BerriAI/litellm/security/advisories/GHSA-jjhc-v7c2-5hh6)

### CVE-2026-35029: Privilege escalation via `/config/update` (High)

Found by Lakera.

`/config/update` didn't check the caller's role. Any authenticated user could modify the proxy's runtime configuration, which could lead to arbitrary file read, admin account takeover, or remote code execution. We now require the `proxy_admin` role on this endpoint.

Full advisory: [GHSA-53mr-6c8q-9789](https://github.com/BerriAI/litellm/security/advisories/GHSA-53mr-6c8q-9789)

### Password hash exposure and pass-the-hash login (High)

Weak hashing originally reported by GitHub user [hamzayevmaqsud](https://github.com/hamzayevmaqsud) ([#15484](https://github.com/BerriAI/litellm/issues/15484)). The full chain was identified by Luca Vandenweghe and Maarten De Rammelaere of [iO Digital](https://www.iodigital.com/).

Passwords were stored as unsalted SHA-256 hashes, and in some cases plaintext. Several API endpoints returned the hash to any authenticated user, and `/v2/login` accepted the raw hash as a credential without re-hashing it, so a stolen hash was as good as the password itself. We've moved to scrypt with random salts and stripped hashes from all API responses.

Full advisory: [GHSA-69x8-hrgq-fjj8](https://github.com/BerriAI/litellm/security/advisories/GHSA-69x8-hrgq-fjj8)

## Bug bounty program

After the supply chain incident and these disclosures it was clear we needed more external eyes on the project. We've set up a bug bounty program so researchers have a way to report issues. 

Bounties are currently paid for P0 (supply chain) and P1 (unauthenticated proxy access) vulnerabilities:

| Severity | Bounty | Example |
|----------|--------|---------|
| Critical | $1,500 – $3,000 | Supply chain compromise |
| High | $500 – $1,500 | Unauthenticated access to protected data |

We plan on expanding the program further in the coming months. More info about the bug bounty program is available [here](https://github.com/BerriAI/litellm/security).

## What's next

Veria Labs is continuing to work with us on a broader audit of the proxy. Security advisories sent through Github will be responded to within five business days. We'll publish advisories as issues are confirmed and fixed.
