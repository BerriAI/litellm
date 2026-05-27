---
slug: may-townhall-updates
title: "May Townhall Updates: Security Hardening, Release Versioning, and the Agent Platform"
date: 2026-05-26T12:00:00
authors:
  - krrish
  - ishaan-alt
description: "A recap of the May LiteLLM town hall covering 89 security fixes, new release versioning, MCP toolsets, performance wins, and the LiteLLM Agent Platform."
tags: [townhall, security, performance, product, agents]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

Thank you to everyone who joined our May town hall.

We covered security hardening, release versioning changes, new product launches (MCP toolsets, on-behalf-of OAuth), performance wins, and our biggest bet yet — the LiteLLM Agent Platform.

{/* truncate */}

## Security updates

### v1.84.1 ships the security hardening

All security fixes from the last 4 weeks are bundled in [v1.84.1](/release_notes/v1.84.1/v1-84-1) — a patch on top of v1.84.0. Upgrade when you can.

```
pip install --upgrade litellm
```

- Backwards-compatible with v1.83.x configs.
- New release versioning scheme (see below).

### Bug bounty — now live

We now pay for security reports.

- **Scope** — the LiteLLM gateway and SDK.
- **Submit** via [private vulnerability report on GitHub](https://github.com/BerriAI/litellm/security).
- **Triaged** by maintainers and Veria Labs security team.

### Automated security review on every PR

Every PR now gets an automated security pass via Veria AI + zizmor + semgrep. Look for the **Veria scan** — it's a required check. False positives are flagged, never blocking.

### Last 4 weeks: by the numbers

| Metric | Count |
|--------|-------|
| Vulnerabilities patched | **89** |
| Reported by Veria scanner | 78 |
| GHSAs fixed | 58 |
| GHSAs closed | 96 |

All fixes ship in [v1.84.1](/release_notes/v1.84.1/v1-84-1).

### What's next for security

- Improve GHSA triage and validation process.
- Further CI pipeline improvements.
- Add zizmor to sister projects (project-releaser).
- Define support window for prior releases.

## Stability updates

### Release versioning — the problem

Too many version suffixes: `-nightly`, `-dev`, `-stable`, `-stable-patch`. Weekly stable bumps left no room for hotfixes, and users filtering `-stable` in search still had to wade through releases.

### New versioning from v1.84.0

Release versions are now consistent across PyPI and Docker.

- **No more `-stable`** — stable releases follow PEP-440 / SemVer 2.0. They now read as `v1.84.0`.
- **Minor bumps weekly** — each scheduled stable release bumps the MINOR version, not PATCH.
- **Patch for hotfixes** — when `v1.84.0` needs a fix, it becomes `v1.84.1`.

### What's next for stability

- EKS multi-pod internal deployment.
- Catch deployment regressions and Claude Code changes.
- Higher code coverage — 70% on 5 hotspot regression files.
- Goal: minimal regressions per stable release.

## Product updates

### What we launched

**Routing & Memory**
- Adaptive Routing
- Memory Management (beta)
- Prompt Compression

**MCP**
- MCP Toolsets
- On-behalf-of MCP OAuth

**Quality & Safety**
- LLM-as-a-judge guardrails
- Skills Marketplace

### MCP Toolsets

MCP Toolsets let you combine tools across multiple MCP servers into a single flat list. An agent sees one tool list instead of juggling multiple servers.

Tools are name-scoped, so collisions across servers are safe.

**Example:** A "deploy-flow" toolset might combine `create_issue` from GitHub MCP, `post_message` from Slack MCP, and `create_ticket` from Jira MCP — all surfaced to the agent as one tool list.

<Image
  img={require('../../img/may_townhall_mcp_toolsets.png')}
  style={{width: '900px', height: 'auto', display: 'block'}}
/>

### MCP on-behalf-of OAuth

OAuth tokens are vaulted at the proxy — never returned to the client.

- The client sends requests without a token.
- LiteLLM adds the token when calling the downstream MCP server.
- Refresh happens transparently. The client never sees a 401.

<Image
  img={require('../../img/may_townhall_mcp_obo_oauth.png')}
  style={{width: '900px', height: 'auto', display: 'block'}}
/>

### What's next for product

- MCP — store static user credentials.
- Claude Code — auto-update header compatibility chart.
- Reasoning level support across models and providers.
- Full Bedrock Converse support in Claude Code.

## Performance wins

### 20% RPS + TPM improvement

Streaming `/chat/completions` now handles 20% more requests per second and tokens per minute.

### Shipped optimizations

<Image
  img={require('../../img/may_townhall_perf_numbers.png')}
  style={{width: '900px', height: 'auto', display: 'block'}}
/>

### What's next for performance

- Rust migration in flight — stable 1K+ RPS at 10k concurrency.
- Focus on reducing gateway overhead under high load.
- Tracking: TTFT, TPM (streaming); RPS, overhead % of E2E (non-streaming).

## Product roadmap: the LiteLLM Agent Platform

### Our bet

We believe 80% of AI workloads will be agents within the next 3 years.

Signals we're seeing:
- OpenClaw usage explosion
- Enterprise asks shifting from chat to agents
- Claude Code adoption tracking up

### LiteLLM Agent Platform — run agents you can actually govern

Four pillars. One control plane.

<Image
  img={require('../../img/may_townhall_agent_platform.png')}
  style={{width: '900px', height: 'auto', display: 'block'}}
/>

- Agent Templates — pre-built configs for common tasks.
- Skills — upload and reuse skills across agents.
- Projects — repos + env vars, packaged for reuse.

## What's next

Thank you again for all the questions and feedback. We'll keep sharing concrete progress updates as these efforts ship.

## Hiring

We are actively hiring across several roles — apply [here](https://jobs.ashbyhq.com/litellm) if you're interested!
