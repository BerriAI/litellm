---
slug: april-townhall-updates
title: "April Townhall Updates: CI/CD v2, Stability, and Product Roadmap"
date: 2026-04-10T12:00:00
authors:
  - krrish
  - ishaan-alt
description: "A recap of the April LiteLLM town hall covering CI/CD v2, product stability work, and the near-term roadmap."
tags: [townhall, security, reliability, product]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

Thank you to everyone who joined our April town hall.

We used the session to share our CI/CD v2 improvements, product stability work, and what we are prioritizing next across reliability and product roadmap.

{/* truncate */}

## CI/CD v2 improvements

Our CI/CD v2 work is centered around four goals:

1. **Limit** what each package can access
2. **Reduce** the number of sensitive environment variables
3. **Avoid** compromised packages
4. **Reduce the risk of** release tampering

#### New architecture: isolated environments

We have begun moving to isolated environments for distinct CI/CD stages to reduce the chance that a single compromised step can inherit broad access across the entire pipeline.

<Image
  img={require('../../img/april_townhall_isolated_environments.png')}
  style={{width: '900px', height: 'auto', display: 'block'}}
/>

#### Current rollout status

These changes are deployed in our current release workflow. [See here](https://github.com/BerriAI/litellm/tags)

#### Independently verify releases

A key part of CI/CD v2 is supporting independent verification of release artifacts using our published verification process, while reducing reliance on any single credential or release path.

[**Learn more about how to verify releases**](https://docs.litellm.ai/docs/proxy/docker_image_security)

<Image
  img={require('../../img/verify_releases.png')}
  style={{width: '900px', height: 'auto', display: 'block'}}
/>

## Stability improvements

### SDLC improvements

This month, we're focusing on process stability improvements around:
- Improving main-branch stability
- Mapping UI QA to built Docker images for 1:1 environment parity
- Consistent release tags across PyPI and Docker
- Fixing release notes publication

#### Improving main-branch stability

We're introducing a staging-gated flow:

<Image
  img={require('../../img/stable_main.png')}
  style={{width: '900px', height: 'auto', display: 'block'}}
/>

- Only an internal staging branch can push to `main`.
- PRs to that staging branch must pass CircleCI LLM API testing.
- Collision handling happens on staging, which is designed to reduce unstable changes reaching `main`.

#### UI QA in Docker environment

Moving forward, all UI QA will be performed in the built Docker image that users run.

Previously, some UI QA paths were run in local environments that did not fully replicate Docker runtime conditions.

That contributed to release-specific issues, including MCP registration problems in `v1.82.3`.

#### Consistent release tags

Today we publish releases for multiple scenarios:
- Dev (Built of a PR for a customer-specific scenario)
- Nightly (Passes all CI/CD checks)
- Release Candidate (Passes all CI/CD checks + manual UI QA)
- Stable (intended to pass all CI/CD checks + manual UI QA + 7 days of production testing)

We are targeting a consistent naming convention across PyPI and Docker by the end of April.

#### Release notes

CI/CD v2 changes moved release notes to a manual path. This is a temporary solution while we investigate a better automated workflow. We are targeting a more consistent process by the end of April.

### Product stability improvements

#### Stable Prisma migrations

Today, we have observed several migration failure classes:
- Migration not applied
- Migration marked applied but incomplete
- Migration not applied due to non-root image issues

We're prioritizing this work this month and have assigned an engineering owner to the effort. Our target is to resolve these error classes by the end of April.

#### UI type safety

Another area of focus is improving the stability of the UI. Today, one cause of errors is that the UI maintains its own assumptions about backend API types. This can lead to issues when backend responses differ from UI assumptions.

We aim to move to having the UI and Backend be in sync with each other, and are exploring OpenAPI-driven mapping to achieve this.

## Product roadmap

### Our Assumptions

Over the next few years, we expect:
- Companies will give employees more AI tools.
- More AI agents will move into production workflows across HR, finance, support, and operations.

### Our Inferences
#### Near-term

- AI spend will increase.
- Uptime and latency will become even more important.
- More AI resources (skills, CLIs, and related assets) will require governance.
- Agent and MCP usage patterns will require deeper controls.
- Broader developer adoption will increase the need for simpler, more discoverable tooling.

#### Long-term 

- We expect many organizations to treat agent auditability (how decisions were made across LLM + MCP + sub-agent inputs/outputs) as a compliance expectation.
- Permission management will get more complex as user-agent interaction chains deepen.

Roadmap timelines in this post are targets and may evolve based on validation and user feedback.

## April investments

### Reliability

- Increase uptime for 10k+ RPS scenarios.
- Investigate latency overhead for long-running Claude Code requests.

### Feature reliability

- Polish MCP authentication.
- Better understand how teams are using agents through LiteLLM.

### Governance

- Launch Skills as a first-class citizen in LiteLLM.

## Q&A

Thank you again for all the questions and direct feedback. We will keep sharing concrete progress updates as these efforts ship.

## Hiring

We are actively hiring across several roles, please apply [here](https://jobs.ashbyhq.com/litellm) if you're interested!