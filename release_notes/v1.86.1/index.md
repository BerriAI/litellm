---
title: "v1.86.1 - Non-Root Docker Build Fix"
slug: "v1-86-1"
date: 2026-05-26T00:00:00
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
docker.litellm.ai/berriai/litellm:1.86.1
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.86.1
```

</TabItem>
</Tabs>

`v1.86.1` is a Dockerfile-only patch on top of [`v1.86.0`](/release_notes/v1.86.0/v1-86-0). The application code is unchanged.

### Infrastructure

- **Docker**
    - Restore `npm` to the `Dockerfile.non_root` builder stage so `prisma-python` resolves Node and no longer falls back to a `nodeenv`-bootstrapped runtime - [PR #28519](https://github.com/BerriAI/litellm/pull/28519)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.86.0...v1.86.1
