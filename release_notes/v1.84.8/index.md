---
title: "v1.84.8 - Database Resilience & Grace-Period Key Rotation"
slug: "v1-84-8"
date: 2026-06-12T18:20:57
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
docker.litellm.ai/berriai/litellm:1.84.8
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.84.8
```

</TabItem>
</Tabs>

`v1.84.8` is a patch release on top of [`v1.84.7`](/release_notes/v1.84.7/v1-84-7). It backports a database-resilience set (Prisma reconnection, prepared-statement and timeout controls, 5xx on DB infra errors during auth) together with the complete grace-period key rotation fix, plus routing and streaming corrections.

### What's Changed

- fix(anthropic): correct streaming reasoning token usage - [PR #27319](https://github.com/BerriAI/litellm/pull/27319)
- fix(proxy): grace-period key rotation deprecated-key lookup - [PR #27756](https://github.com/BerriAI/litellm/pull/27756)
- fix(router): use forwarded model_id for native Azure container IDs - [PR #27921](https://github.com/BerriAI/litellm/pull/27921)
- fix(proxy): recover from cached-plan errors by reconnecting the Prisma client - [PR #29983](https://github.com/BerriAI/litellm/pull/29983)
- fix(proxy): expose Prisma idle/connect timeout and extra DB URL params - [PR #28395](https://github.com/BerriAI/litellm/pull/28395)
- feat(proxy): add option to disable server-side prepared statements for DB lookups - [PR #29984](https://github.com/BerriAI/litellm/pull/29984)
- fix(proxy): return 5xx on DB infra errors during auth - [PR #29986](https://github.com/BerriAI/litellm/pull/29986)
- fix(passthrough): resolve costing model when body model is unknown - [PR #30160](https://github.com/BerriAI/litellm/pull/30160)
- fix(proxy): return deprecated-key lookup result directly in get_data combined view - [PR #30327](https://github.com/BerriAI/litellm/pull/30327)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.84.7...v1.84.8
