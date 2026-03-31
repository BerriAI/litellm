---
slug: ci-cd-v2-improvements
title: "Announcing CI/CD v2 for LiteLLM"
date: 2026-03-30T21:30:00
authors:
  - krrish
description: "CI/CD v2 introduces isolated environments, stronger security gates, and safer release separation for LiteLLM."
tags: [engineering, ci-cd, security]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

The CI/CD v2 is now live for LiteLLM.

<Image
  img={require('../../img/ci_cd_architecture.png')}
  style={{width: '700px', height: 'auto', display: 'block'}}
/>

<br/>
Building on the roadmap from our [security incident](https://docs.litellm.ai/blog/security-townhall-updates#roadmap), CI/CD v2 introduces isolated environments, stronger security gates, and safer release separation for LiteLLM.

## What changed

- Security scans and unit tests run in isolated environments.
- Validation and release are separated into different repositories, making it harder for an attacker to reach release credentials.
- Trusted Publishing for PyPI releases - this means no long-lived credentials are used to publish releases.
- Immutable Docker release tags - this means no tampering of Docker release tags after they are published [Learn more](https://docs.docker.com/docker-hub/repos/manage/hub-images/immutable-tags/). Note: work for GHCR docker releases is planned as well.

## What's next

Moving forward, we plan on:
- Adopting OpenSSF (this is a set of security criteria that projects should meet to demonstrate a strong security posture - [Learn more](https://baseline.openssf.org/versions/2026-02-19.html))
  - We've added Scorecard and Allstar to our Github

- Adding SLSA Build Provenance to our CI/CD pipeline - this means we allow users to independently verify that a release came from us and prevent silent modifications of releases after they are published.


We hope that this will mean you can be confident that the releases you are using are safe and from us.


## The principle

The new CI/CD pipeline reflects the principles, outlined below, and is designed to be more secure and reliable:

- **Limit** what each package can access
- **Reduce** the number of sensitive environment variables
- **Avoid** compromised packages
- **Prevent** release tampering


## How to help: 

Help us plan April's stability sprint - https://github.com/BerriAI/litellm/issues/24825