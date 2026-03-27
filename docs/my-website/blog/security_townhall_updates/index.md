---
slug: security-townhall-updates
title: "Security Townhall Updates"
date: 2026-03-27T12:00:00
authors:
  - krrish
  - ishaan-alt
description: "What happened, what we've done, and what comes next for LiteLLM's release and security processes."
tags: [security, incident-report]
hide_table_of_contents: false
---

Thank you to everyone who joined our town hall.

We wanted to use that time to walk openly through what we know, what happened, what we've done so far, and how we're improving LiteLLM's release and security processes going forward. This post is a written version of that update.

{/* truncate */}

## What happened

On March 24, 2026 at 10:39 UTC, LiteLLM v1.82.7 was pushed to PyPI. Version v1.82.8 was published soon after. Those packages were live for about 40 minutes before being quarantined by PyPI. By 16:00 UTC, the LiteLLM team had worked with PyPI to delete the affected packages.

At this point, our understanding is that this was a supply-chain incident affecting those two published versions.

### Were older packages impacted?

Our current findings show no indicators of compromise in the last 20 versions of LiteLLM. This was manually verified by our team and independently reviewed by Veria Labs.

We have also published a new verified safe version for users to upgrade to.

## How did this happen?

Our current understanding is that the root issue came from a compromised dependency in our CI/CD pipeline.

There were three major contributing factors:

### 1. Shared CI/CD environment

At the time, everything was running on CircleCI, and all steps shared a common environment. That increased blast radius: if one component was compromised, it could potentially access credentials or context intended for other parts of the pipeline.

### 2. Static credentials in environment variables

Release credentials, including credentials for PyPI, GHCR, and Docker publishing, were available as static secrets in the environment. That meant a compromised step could potentially access long-lived release credentials.

### 3. Unpinned Trivy dependency

In our security scanning component, we had an unpinned Trivy dependency. Our present understanding is that a compromised Trivy package ran during the scan, had access to environment variables, and enabled attackers to obtain those credentials.

In plain terms: a compromised package in CI had access to secrets it should not have had, and those secrets were then used in the release path.

## What we've already done

Since the incident, we have taken immediate steps across credentials, repository hygiene, code verification, and release hardening.

### Prevented further key abuse

We deleted or rotated all impacted secret keys, including PyPI, GitHub, Docker, and related credentials. We also rotated LiteLLM maintainer accounts.

### Reduced repository attack surface

We removed roughly 6,000 open branches and added an auto-deletion policy for branches merged into `main`. This reduces the surface area for branch-based abuse and keeps the repo easier to reason about during incident response.

### Paused releases

We paused new releases until we could confirm codebase security and put stronger release controls in place.

### Verified the codebase

We are working with Google's Mandiant cybersecurity team to confirm the source of the attack and verify the security of the codebase. We also confirmed that no malicious code was pushed to `main`.

In parallel, we are working with whitehat hackers at Veria Labs to verify application security and review improvements to our CI/CD process.

We have also confirmed that the last 20 LiteLLM releases contain no indicators of compromise, and that no unauthenticated attacks can be made against LiteLLM Proxy based on our current investigation.

### Created a security working group

We created a new security working group inside LiteLLM focused on:

- Building threat models
- Auditing the build process and dependencies
- Reviewing release pipeline changes before rollout

## CI/CD improvements already underway

We are making structural changes to how releases are built and published. These changes focus on reducing credential exposure, preventing dependency-based compromise, and making releases auditable and tamper-evident.

### 1. Reducing static credentials

We are setting up PyPI Trusted Publishing on GitHub Actions so releases can use short-lived, identity-based credentials instead of long-lived static secrets.

The goal is simple: even if one part of CI is compromised, there should not be a reusable publishing credential sitting in an environment variable.

### 2. Preventing unpinned dependencies

We have pinned GitHub Actions and are working on pinning all CircleCI dependencies as well. We are also setting up Zizmor to harden GitHub Actions and catch issues such as unpinned dependencies and credential leakage.

### 3. Ensuring immutable, auditable releases

We are working on Cosign keyless signing for both PyPI and GHCR releases. This will allow users to independently verify that a release came from us and help ensure published artifacts cannot be silently modified later.

This is intended to protect against:

- Stolen PyPI or GHCR credentials
- Tampered registry artifacts
- Tag mutation

## Our roadmap going forward

We are using four guiding principles to redesign the release pipeline and improve our security posture.

### 1. Limit what each package can access

A package used in one step of CI should not automatically have access to the entire environment. We want each component to see only the minimum set of variables and permissions it needs.

### 2. Reduce the number of sensitive environment variables

Especially for release and publishing systems, we want fewer standing secrets in the environment overall.

### 3. Avoid compromised packages

We are moving toward pinned, verified SHAs for packages and actions used in CI/CD, avoiding `latest` wherever possible, and adding a cooldown period before dependency upgrades are trusted in release-critical paths.

### 4. Prevent release tampering

Every release should be attributable, auditable, and independently verifiable. That means ephemeral credentials, artifact signing, and clearer release provenance.

## Architectural direction: isolating environments

One of the most important shifts is separating environments by function. Instead of one shared environment, we are moving toward distinct pipelines for:

- Unit tests
- Integration tests
- Security scans
- Release publishing

This limits the damage that any single compromised component can cause.

## What this means for users

The key takeaways are:

- The malicious packages were live for about 40 minutes
- The supply-chain attack appears to have originated from a compromised Trivy security scanner dependency
- `main` is safe based on our current investigation
- The last 20 releases show no indicators of compromise
- We have formed a dedicated security working group
- Our new CI/CD direction is centered on isolated environments, ephemeral credentials, and release auditing

## Closing

We know incidents like this affect trust, and trust has to be earned back through transparency and concrete action.

Our focus now is not just fixing the immediate problem, but building a safer release system with much tighter boundaries: isolated environments, fewer secrets, stronger dependency controls, and verifiable releases.

We'll continue to share updates as we complete the RCA and roll out the next set of security improvements.

---

For real-time updates, follow [LiteLLM (YC W23) on X](https://x.com/LiteLLM). If you have questions, reach out at `security@berri.ai` or `support@berri.ai`.
