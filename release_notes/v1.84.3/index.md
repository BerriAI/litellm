---
title: "v1.84.3 - Dockerfile Re-cut for Non-Root Image"
slug: "v1-84-3"
date: 2026-05-27T00:00:00
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
docker.litellm.ai/berriai/litellm:1.84.3
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.84.3
```

</TabItem>
</Tabs>

`v1.84.3` is a Dockerfile-only re-cut of [`v1.84.2`](/release_notes/v1.84.2/v1-84-2); the application code is identical. It restores `npm` to the `Dockerfile.non_root` builder stage so the `litellm-non_root:1.84.3` image builds, which the `1.84.2` image did not.

If you are upgrading from [`v1.84.1`](/release_notes/v1.84.1/v1-84-1), see the [`v1.84.2`](/release_notes/v1.84.2/v1-84-2) notes for the underlying code changes; in particular the path-handling hardening covered in the [host-header authentication bypass advisory](/blog/host-header-auth-bypass).

## Full Changelog

https://github.com/BerriAI/litellm/compare/5560f35279...v1.84.3
