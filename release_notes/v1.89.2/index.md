---
title: "v1.89.2 - Cost Tracking & Model-List Fixes"
slug: "v1-89-2"
date: 2026-06-17T19:22:38
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
docker.litellm.ai/berriai/litellm:1.89.2
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.89.2
```

</TabItem>
</Tabs>

`v1.89.2` is a patch release on top of [`v1.89.1`](/release_notes/v1.89.1/v1-89-1). It hardens cost tracking around `service_tier`, corrects `/v1/models` listing for team and BYOK setups, and tightens vector-store access and OTEL error reporting.

### What's Changed

- fix(cost): stop a non-string `service_tier` from silently dropping cost tracking - [PR #30690](https://github.com/BerriAI/litellm/pull/30690)
- fix(anthropic): price and surface the response `service_tier` in cost tracking - [PR #30558](https://github.com/BerriAI/litellm/pull/30558)
- fix(proxy): list the public team model name in `/v1/models` - [PR #30588](https://github.com/BerriAI/litellm/pull/30588)
- feat(proxy): add an opt-in `healthy_only` filter to `GET /v1/models` - [PR #30130](https://github.com/BerriAI/litellm/pull/30130)
- fix(proxy): resolve list-files credentials from team BYOK deployments - [PR #30495](https://github.com/BerriAI/litellm/pull/30495)
- fix(proxy): allow internal roles to access vector store CRUD routes - [PR #30503](https://github.com/BerriAI/litellm/pull/30503)
- fix(otel): record the full error message on the standard exception event in OTEL v2 - [PR #30380](https://github.com/BerriAI/litellm/pull/30380)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.89.1...v1.89.2
