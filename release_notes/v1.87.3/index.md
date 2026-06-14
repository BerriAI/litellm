---
title: "v1.87.3 - DB Resilience & Passthrough Hardening"
slug: "v1-87-3"
date: 2026-06-13T17:37:18
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
docker.litellm.ai/berriai/litellm:1.87.3
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.87.3
```

</TabItem>
</Tabs>

`v1.87.3` is a patch release on top of [`v1.87.2`](/release_notes/v1.87.2/v1-87-2). It brings the 1.84.8 database-resilience set onto the 1.87 line, adds the budget-reservation toggle, hardens Anthropic streaming logging, and refreshes dependencies.

### What's Changed

- feat(proxy): add `disable_budget_reservation` general setting - [PR #29493](https://github.com/BerriAI/litellm/pull/29493)
- fix(proxy): recover from cached-plan errors by reconnecting the Prisma client - [PR #29983](https://github.com/BerriAI/litellm/pull/29983)
- feat(proxy): add option to disable server-side prepared statements for DB lookups - [PR #29984](https://github.com/BerriAI/litellm/pull/29984)
- fix(proxy): return 5xx on DB infra errors during auth; reserve 401 for genuine auth failures - [PR #29986](https://github.com/BerriAI/litellm/pull/29986)
- fix(passthrough): resolve costing model when body model is unknown - [PR #30160](https://github.com/BerriAI/litellm/pull/30160)
- fix(passthrough): skip `[DONE]` sentinels and non-JSON SSE frames in Anthropic streaming logging - [PR #30202](https://github.com/BerriAI/litellm/pull/30202)
- fix(proxy): return deprecated-key lookup result directly in get_data combined view - [PR #30327](https://github.com/BerriAI/litellm/pull/30327)
- chore(deps): bump pypdf, tornado, the aiohttp constraint, vitest, and brace-expansion - [PR #30220](https://github.com/BerriAI/litellm/pull/30220)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.87.2...v1.87.3
