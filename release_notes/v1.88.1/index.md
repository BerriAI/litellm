---
title: "v1.88.1 - Dependency Bumps"
slug: "v1-88-1"
date: 2026-06-08T17:23:56
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
docker.litellm.ai/berriai/litellm:1.88.1
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.88.1
```

</TabItem>
</Tabs>

`v1.88.1` is a patch release on top of [`v1.88.0`](/release_notes/v1.88.0/v1-88-0). It bumps PyJWT and the `ws` override to clear dependency advisories on the 1.88 line.

### What's Changed

- build(deps): bump PyJWT to 2.13.0 and the `ws` override to 8.20.1 - [PR #29987](https://github.com/BerriAI/litellm/pull/29987)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.88.0...v1.88.1
