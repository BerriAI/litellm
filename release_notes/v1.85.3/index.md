---
title: "v1.85.3 - Observability, Budget & Rate-Limit Fixes"
slug: "v1-85-3"
date: 2026-06-01T19:02:53
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
docker.litellm.ai/berriai/litellm:1.85.3
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.85.3
```

</TabItem>
</Tabs>

`v1.85.3` is a patch release on top of [`v1.85.2`](/release_notes/v1.85.2/v1-85-2). It cherry-picks fixes for duplicate Claude Code traces, Bearer-prefix hashing, budget-reset writes, and two flag-leak corrections in the rate limiter and the provider request body.

### What's Changed

- fix(logging): stop duplicate Claude Code traces (internal copy of #29089) - [PR #29311](https://github.com/BerriAI/litellm/pull/29311)
- fix(proxy): normalize the Bearer prefix in the safe-hash helper - [PR #29343](https://github.com/BerriAI/litellm/pull/29343)
- fix(budget): reset_budget writes only `{spend, budget_reset_at}` and no longer pre-zeroes the counter - [PR #29358](https://github.com/BerriAI/litellm/pull/29358)
- fix(rate-limit): stop the v3 limiter from leaking internal stash to the provider body - [PR #27913](https://github.com/BerriAI/litellm/pull/27913)
- fix(proxy): stop the `use_chat_completions_api` flag from leaking into the provider request body - [PR #29447](https://github.com/BerriAI/litellm/pull/29447)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.85.2...v1.85.3
