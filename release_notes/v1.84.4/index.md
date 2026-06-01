---
title: "v1.84.4 - Reset-Budget & Observability Fixes"
slug: "v1-84-4"
date: 2026-05-31T00:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Yuneng Jiang
    title: Senior Full Stack Engineer, LiteLLM
    url: https://www.linkedin.com/in/yuneng-david-jiang-455676139/
    image_url: https://avatars.githubusercontent.com/u/171294688?v=4
hide_table_of_contents: false
---

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="docker" label="Docker">

```bash
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:1.84.4
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.84.4
```

</TabItem>
</Tabs>

`v1.84.4` is a patch release on top of [`v1.84.3`](/release_notes/v1.84.3/v1-84-3). It fixes a `ResetBudgetJob` regression that affected UI-created keys on the `v1.84` line, removes duplicate observability exports for Claude Code streams, and tightens a Bearer-prefix edge case in the proxy auth metric labels.

### Bug Fixes

- **Spend Tracking, Budgets and Rate Limiting**
    - `ResetBudgetJob` no longer pre-zeroes the spend counter and only writes `{spend, budget_reset_at}` on cycle rollover. On `v1.84.0+` this previously silently failed for every UI-created key, leaving spend frozen at the cycle boundary and opening a brief budget-enforcement bypass window on every scheduler tick - [PR #29358](https://github.com/BerriAI/litellm/pull/29358)

- **Logging / Observability**
    - Stop emitting duplicate Datadog logs and OTLP traces (Arize Phoenix, Langfuse, etc.) on completed streams by dispatching the success handler exactly once instead of running both `async_success_handler` and `success_handler` - [PR #29311](https://github.com/BerriAI/litellm/pull/29311) (originally [PR #29089](https://github.com/BerriAI/litellm/pull/29089))

- **Proxy auth**
    - Normalize the `Bearer ` prefix in the safe-hash helper so the failure-metric label is hashed identically whether or not the caller stripped the prefix - [PR #29343](https://github.com/BerriAI/litellm/pull/29343)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.84.3...v1.84.4
