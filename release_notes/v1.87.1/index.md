---
title: "v1.87.1 - Azure AD, Batch Auth & Key Access Backports"
slug: "v1-87-1"
date: 2026-06-03T21:54:19
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
docker.litellm.ai/berriai/litellm:1.87.1
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.87.1
```

</TabItem>
</Tabs>

`v1.87.1` is a patch release on top of [`v1.87.0`](/release_notes/v1.87.0/v1-87-0). It backports five staged fixes: Azure AD token refresh, batch and video model routing, org-scoped team key creation, and Vertex Claude effort handling. The duplicate passthrough cost-callback fix the other lines received is deliberately held back here, since 1.87.x branched separately and carries a different passthrough logging path where that guard does not exist.

### What's Changed

- fix(azure): preserve AD token refresh in the v1 OpenAI client path - [PR #28627](https://github.com/BerriAI/litellm/pull/28627)
- fix(proxy): map a stripped batch `body.model` back to the proxy alias so key access checks pass - [PR #29264](https://github.com/BerriAI/litellm/pull/29264)
- fix(proxy): resolve managed video model ids through the router before auth, budget, and key checks - [PR #29545](https://github.com/BerriAI/litellm/pull/29545)
- fix(key_generate): let team members create keys on org-scoped teams (regression since v1.84.0-rc.1) - [PR #29310](https://github.com/BerriAI/litellm/pull/29310)
- fix(vertex): strip `output_config.effort` for Vertex Claude models that reject it, such as Haiku 4.5 - [PR #29585](https://github.com/BerriAI/litellm/pull/29585)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.87.0...v1.87.1
