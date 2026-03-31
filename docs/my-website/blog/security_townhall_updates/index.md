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

import Image from '@theme/IdealImage';

Thank you to everyone who joined our town hall.

We wanted to use that time to walk through what we know, what we've done so far, and how we're improving LiteLLM's release and security processes going forward. This post is a written version of that update. [Slides available here](https://drive.google.com/file/d/17hsSG7nk-OYL7VRCTbTa7McrWREtS9OO/view?usp=sharing)

{/* truncate */}

## What happened

On March 24, 2026 at 10:39 UTC, LiteLLM v1.82.7 was pushed to PyPI. Version v1.82.8 was published soon after. Those packages were live for about 40 minutes before being quarantined by PyPI. By 16:00 UTC, the LiteLLM team had worked with PyPI to delete the affected packages.

At this point, our understanding is that this was a supply-chain incident affecting those two published versions.

## How did this happen?

Our understanding is that the issue came from the [compromised Trivy security scanner](https://www.aquasec.com/blog/trivy-supply-chain-attack-what-you-need-to-know/) dependency in our CI/CD pipeline.

<Image 
  img={require('../../img/shared_ci_cd_environment.png')}
  style={{width: '500px', height: '400px', display: 'block'}}
/>

There were three major contributing factors:

### 1. Shared CI/CD environment

At the time, everything was running on CircleCI, and all steps shared a common environment. That increased blast radius: if one component was compromised, it could potentially access credentials or context intended for other parts of the pipeline.

### 2. Static credentials in environment variables

Release credentials, including credentials for PyPI, GHCR, and Docker publishing, were available as static secrets in the environment. That meant a compromised step could access long-lived release credentials.

### 3. Unpinned Trivy dependency

In our security scanning component, we had an unpinned Trivy dependency. Our present understanding is that a compromised Trivy package ran during the scan, had access to environment variables, and enabled attackers to obtain those credentials.

**In summary:** a compromised package in CI had access to secrets it should not have had, and those secrets were then used in the release path.

## What we've already done


In the last 3 days, we've taken the following steps:

### 1. Minimize Scope of Impact

#### Prevented further key abuse

We deleted or rotated all impacted or adjacent secret keys, including PyPI, GitHub, Docker, and related credentials. Out of an abundance of caution, we've also rotated LiteLLM maintainer accounts. 

#### Prevent branch attacks

We removed roughly 6,000 open branches and added an auto-deletion policy for branches merged into `main`. This reduces the surface area for branch-based abuse.

#### Pinned CI/CD dependencies

We've pinned all Github Actions, and are working on pinning all CircleCI dependencies as well.

#### Paused releases

We've paused new releases until we've confirmed codebase security and put stronger release controls in place.

### 2. Secured LiteLLM

#### Forensic analysis

We are working with Google's Mandiant cybersecurity team to confirm the source of the attack and verify the security of the codebase. We also confirmed that no malicious code was pushed to `main`.

#### Confirm Application Security

In parallel, we are working with whitehat hackers at [Veria Labs](https://verialabs.com/) to verify application security and review improvements to our CI/CD process.

We have also confirmed that the last 20 LiteLLM releases contain no indicators of compromise, and that no unauthenticated attacks can be made against LiteLLM Proxy based on our current investigation. [Check Security Blog for release verification.](https://docs.litellm.ai/blog/security-update-march-2026#verified-safe-versions)

#### Created a security working group

We created a new security working group inside LiteLLM focused on:

- Building threat models
- Auditing the build process and dependencies

If you're interested in joining the security working group, please file an issue [here](https://github.com/BerriAI/litellm-security-wg).

### 3. Improved CI/CD

We've already begun making structural changes to how releases are built and published. These align with our goals (covered in the next section) around isolated environments, ephemeral credentials, and release auditing.

## Roadmap

We plan on following 4 guiding principles for our new CI/CD pipeline:

1. **Limit** what each package can access
2. **Reduce** the number of sensitive environment variables
3. **Avoid** compromised packages
4. **Prevent** release tampering


### Isolated environments

<Image 
  img={require('../../img/isolated_ci_cd_environments.png')}
  style={{width: '400px', height: 'auto'}}
/>

We are breaking our CI/CD into 4 semantic concepts:

1. Unit tests
2. Integration tests
3. Security scans
4. Release publishing

And will be running each of these in isolated environments.

This will limit the damage that any single compromised component can cause.

### Ephemeral credentials

We plan to move to ephemeral credentials for PyPI (Trusted Publisher) and GHCR (Token-based authentication) releases. This will reduce the risk of credentials being leaked or compromised.

We have already begun doing this: 

- PyPI Trusted Publisher on GitHub Actions [PR](https://github.com/BerriAI/litellm/pull/24654)
- GHCR Token-based authentication on GitHub Actions [PR](https://github.com/BerriAI/litellm/pull/24683)

### Release auditing

Our goal is to allow users to independently verify that a release came from us and prevent silent modifications of releases after they are published.

This will ensure, your releases are safe, even when: 
- Stolen PyPI/GHCR credentials are used to publish malicious releases
- Tampered registry artifacts are published
- Tag mutations are made after the release is published

We believe that [Cosign](https://github.com/sigstore/cosign) is a good fit for this, and have already begun working on it [PR](https://github.com/BerriAI/litellm/pull/24683).


### Avoid Compromised Packages

- Move to pinned, verified SHAs for packages and actions used in CI/CD, avoiding `latest` wherever possible. 
- Add a cooldown period before upgrading to a new version of a package - allows more time to investigate and verify the new version. 

We've added zizmor to help us catch issues such as unpinned dependencies and credential leakage. [commit](https://github.com/BerriAI/litellm/commit/a671275f5c5b0e1fb1adacdf3b6ef779aaa5d56c).


## Frequently Asked Questions

**Q: Did you observe any lateral movement into your corporate environment during this incident?**

A: No. Our investigation to date, conducted in coordination with external security experts, has found no evidence of lateral movement into our internal corporate systems. The incident was isolated to the CI/CD pipeline and the release path for specific versions (v1.82.7 and v1.82.8). As a proactive measure, we have rotated all potentially impacted or adjacent secrets—including PyPI, GitHub, and Docker credentials—and updated maintainer account security to ensure continued isolation.

**Q: Do you expect delays in future product releases due to these new security measures?**

A: We are committed to balancing security with speed. While we have temporarily paused releases to implement stronger controls, we are moving quickly to automate our new security protocols. We are currently implementing isolated CI/CD environments, ephemeral credentials (via Trusted Publishers), and release auditing with Cosign. These improvements are designed to be integrated into our automated pipeline, allowing us to maintain a fast release cadence while ensuring every package is verified and secure.

**Q: Were older packages impacted?**

Our current findings show no indicators of compromise in the last 20 versions of LiteLLM. This was manually verified by our team and independently reviewed by Veria Labs.

We have also published the verified versions for users to use. [Check Security Blog for release verification.](https://docs.litellm.ai/blog/security-update-march-2026#verified-safe-versions)



## Questions & Support 

If you believe your systems may be affected, contact us immediately:

- **Security:** security@berri.ai
- **Support:** support@berri.ai
- **Slack:** Reach out to the LiteLLM team directly [here](https://join.slack.com/t/litellmossslack/shared_invite/zt-3o7nkuyfr-p_kbNJj8taRfXGgQI1~YyA)

## Hiring 

We are currently hiring for: 

- DevOps Engineer - to keep ci/cd secure and running smoothly
- Security Engineer - to keep the application secure

If you're interest in joining, please apply [here](https://jobs.ashbyhq.com/litellm)