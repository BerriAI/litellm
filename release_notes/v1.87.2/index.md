---
title: "v1.87.2 - Claude Fable 5, Batch Auth, CrowdStrike & Mantle SigV4"
slug: "v1-87-2"
date: 2026-06-10T21:47:54
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
docker.litellm.ai/berriai/litellm:1.87.2
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.87.2
```

</TabItem>
</Tabs>

`v1.87.2` is a patch release on top of [`v1.87.1`](/release_notes/v1.87.1/v1-87-1). It adds Claude Fable 5, batch-file authorization, the CrowdStrike AIDR identity pair, and SigV4/IAM auth for the Bedrock Mantle Responses API route.

### What's Changed

- feat: add Claude Fable 5 across Anthropic, Bedrock, Vertex AI, and Azure AI - [PR #30064](https://github.com/BerriAI/litellm/pull/30064)
- fix(proxy): authorize batch files using upload `target_model_names` (LIT-3593) - [PR #30009](https://github.com/BerriAI/litellm/pull/30009)
- feat(guardrails): capture CrowdStrike AIDR user and model metadata - [PR #29517](https://github.com/BerriAI/litellm/pull/29517)
- fix(guardrails): read CrowdStrike AIDR identity from both metadata bags - [PR #29991](https://github.com/BerriAI/litellm/pull/29991)
- feat(bedrock_mantle): add SigV4/IAM auth to the Responses API route, with its prerequisite Mantle Responses route (#29490) - [PR #29788](https://github.com/BerriAI/litellm/pull/29788)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.87.1...v1.87.2
