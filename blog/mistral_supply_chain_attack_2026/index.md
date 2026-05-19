---
slug: mistral-supply-chain-attack-may-2026
title: "Security Update: Mistral AI PyPI Supply Chain Attack — LiteLLM Not Impacted"
date: 2026-05-12T10:00:00
authors:
  - mubashir
description: "On May 11, 2026, a malicious version of the mistralai PyPI package was published as part of a coordinated supply chain attack. LiteLLM is not affected — we call Mistral exclusively via httpx, never by importing the mistralai SDK."
tags: [security, supply-chain, mistral, incident-report]
hide_table_of_contents: false
---

On May 11, 2026, security researchers at [Aikido Security](https://www.aikido.dev/blog/mini-shai-hulud-is-back-tanstack-compromised) discovered a coordinated supply chain attack dubbed **"Mini Shai-Hulud"** that published malicious versions of over 170 npm packages and 2 PyPI packages, including `mistralai==2.4.6`. 

**LiteLLM is not impacted.** We call Mistral's API directly over HTTP via `httpx` and do not import the `mistralai` Python SDK anywhere in the codebase.

## TLDR;

- **LiteLLM does not install or import the `mistralai` package.** We call Mistral's API the same way we call every other provider (via `httpx`). The compromised package is never executed in any LiteLLM environment.
- **No LiteLLM user credentials were at risk from this attack.** The malware runs at `import mistralai` time. Since LiteLLM never reaches that import, the payload never fires.
- **No action is required from LiteLLM users.** If you have separately installed `mistralai==2.4.6` in the same environment for your own application code, you should follow [Mistral AI's guidance](https://docs.mistral.ai/resources/security-advisories) immediately.

{/* truncate */}

---

## What happened

TeamPCP published `mistralai==2.4.6` to PyPI — a version Mistral AI never released. The package contained a backdoor injected into `src/mistralai/client/__init__.py` that fires at import time on Linux hosts. When triggered, it downloads a file named `transformers.pyz` from a hardcoded attacker-controlled IP address (`83.142.209.194`) and executes it as a detached background process.

The filename was deliberately chosen to resemble Hugging Face's widely used `transformers` library, giving it cover in ML environments.

The payload functions as a **credential stealer**, targeting secrets stored on the host — cloud credentials, CI/CD tokens, GitHub access tokens, and API keys. Researchers also found a geofenced destructive branch with a 1-in-6 probability of running `rm -rf /` on systems detected to be in certain regions.

PyPI has since quarantined the entire `mistralai` project. The attack was part of a broader campaign that hit TanStack (42 packages), UiPath (65 packages), Guardrails AI, OpenSearch, and others across both npm and PyPI.

---

## What to check if you use LiteLLM

No LiteLLM-specific action is needed. If you want to be thorough:

1. **Confirm `mistralai` is not in your environment.**
   ```bash
   pip show mistralai
   ```
   If the output shows version `2.4.6`, remove it immediately and follow [Mistral AI's security advisory](https://docs.mistral.ai/resources/security-advisories).

2. **Check your environment for the dropper.**
   Look for `/tmp/transformers.pyz` on any Linux hosts that had `mistralai==2.4.6` installed, and for unexpected outbound connections to `83.142.209.194`.

3. **Rotate credentials if you were affected.**
   If `mistralai==2.4.6` was installed and imported in your environment, treat all secrets present on that host as compromised: cloud credentials, API keys, CI/CD tokens, and GitHub tokens.

---

## Our broader approach to dependency security

If you discover a security issue in LiteLLM, please report it through our [bug bounty program](https://github.com/BerriAI/litellm/security). We pay out for P0 (supply chain) and P1 (unauthenticated proxy access) issues.

---

**References**

- [Aikido Security — Mini Shai-Hulud Is Back](https://www.aikido.dev/blog/mini-shai-hulud-is-back-tanstack-compromised)
- [The Hacker News — Mini Shai-Hulud Worm Compromises TanStack, Mistral AI, Guardrails AI & More](https://thehackernews.com/2026/05/mini-shai-hulud-worm-compromises.html)
- [Wiz Blog — Mini Shai-Hulud Strikes Again](https://www.wiz.io/blog/mini-shai-hulud-strikes-again-tanstack-more-npm-packages-compromised)
- [Mistral AI Security Advisories](https://docs.mistral.ai/resources/security-advisories)
- [GitHub Issue #523 — mistralai/client-python](https://github.com/mistralai/client-python/issues/523)
- [SafeDep — Mass Supply Chain Attack Hits TanStack, Mistral AI](https://safedep.io/mass-npm-supply-chain-attack-tanstack-mistral/)
