---
title: "v1.86.3 - Gemini 3.5 Flash Day-0 & Pending Line Backports"
slug: "v1-86-3"
date: 2026-06-02T17:31:21
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
docker.litellm.ai/berriai/litellm:1.86.3
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.86.3
```

</TabItem>
</Tabs>

`v1.86.3` is a patch release on top of [`v1.86.2`](/release_notes/v1.86.2/v1-86-2). It closes the gap with the 1.84 and 1.85 lines: day-0 Gemini 3.5 Flash on Vertex AI and Google AI Studio with its paired Vertex tool-call fix, Redis spend-counter seeding, and the observability, budget, and flag-leak fixes.

### What's Changed

- feat: day-0 support for Gemini 3.5 Flash on Vertex AI and Google AI Studio - [PR #28268](https://github.com/BerriAI/litellm/pull/28268)
- fix(vertex): omit the function_call `id` on Gemini 3.5+ tool turns (pairs with #28268) - [PR #28324](https://github.com/BerriAI/litellm/pull/28324)
- fix(spend): seed the Redis spend counter with `SET NX` so concurrent pods no longer double-seed - [PR #27854](https://github.com/BerriAI/litellm/pull/27854)
- fix(logging): stop duplicate Claude Code traces, plus the `_build_passthrough_logging_result` helper - [PR #29311](https://github.com/BerriAI/litellm/pull/29311)
- fix(proxy): normalize the Bearer prefix in the safe-hash helper - [PR #29343](https://github.com/BerriAI/litellm/pull/29343)
- fix(budget): reset_budget writes only `{spend, budget_reset_at}` - [PR #29358](https://github.com/BerriAI/litellm/pull/29358)
- fix(proxy): stop the `use_chat_completions_api` flag from leaking into the provider request body - [PR #29447](https://github.com/BerriAI/litellm/pull/29447)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.86.2...v1.86.3
