---
title: "v1.89.1 - DB Resilience, MCP & Model-Info Backports"
slug: "v1-89-1"
date: 2026-06-15T20:08:03
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
docker.litellm.ai/berriai/litellm:1.89.1
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.89.1
```

</TabItem>
</Tabs>

`v1.89.1` is a patch release on top of [`v1.89.0`](/release_notes/v1.89.0/v1-89-0). It brings the 1.84.8 database-resilience set onto the 1.89 line, hardens the MCP OAuth and credential paths, aligns the `/v1/model/info` and `/v2/model/info` surfaces, adds the budget-reservation toggle, and refreshes dependencies.

### What's Changed

- fix(proxy): recover from cached-plan errors by reconnecting the Prisma client - [PR #29983](https://github.com/BerriAI/litellm/pull/29983)
- feat(proxy): add option to disable server-side prepared statements for DB lookups - [PR #29984](https://github.com/BerriAI/litellm/pull/29984)
- fix(proxy): return 5xx on DB infra errors during auth; reserve 401 for genuine auth failures - [PR #29986](https://github.com/BerriAI/litellm/pull/29986)
- feat(proxy): add `disable_budget_reservation` general setting - [PR #29493](https://github.com/BerriAI/litellm/pull/29493)
- fix(proxy): align `/v1/model/info` with router deployments - [PR #30025](https://github.com/BerriAI/litellm/pull/30025)
- fix(proxy): populate `access_via_team_ids` on `/v1/model/info` - [PR #30274](https://github.com/BerriAI/litellm/pull/30274)
- feat(proxy): publish `/v2/model/info` in the Swagger OpenAPI spec - [PR #29900](https://github.com/BerriAI/litellm/pull/29900)
- fix(proxy): return deprecated-key lookup result directly in the `get_data` combined view - [PR #30327](https://github.com/BerriAI/litellm/pull/30327)
- fix(mcp): drop phantom 401 span on delegated OAuth2 tool calls - [PR #30494](https://github.com/BerriAI/litellm/pull/30494)
- fix(mcp): drop orphaned per-user credential rows when an MCP server is deleted - [PR #30141](https://github.com/BerriAI/litellm/pull/30141)
- fix(mcp): allow team access-group grants in the OAuth authorize/token access check - [PR #30041](https://github.com/BerriAI/litellm/pull/30041)
- fix(ui/mcp): reset OAuth state on create-server modal close so a prior server's token no longer leaks into the next add-server session - [PR #30000](https://github.com/BerriAI/litellm/pull/30000)
- fix(mcp): load MCP tool configuration tools via the OBO/passthrough-aware GET path - [PR #29960](https://github.com/BerriAI/litellm/pull/29960)
- fix(mcp): let non-creator users OAuth into OBO-mode MCP servers from the Tools page - [PR #29867](https://github.com/BerriAI/litellm/pull/29867)
- fix(passthrough): resolve costing model when body model is unknown - [PR #30160](https://github.com/BerriAI/litellm/pull/30160)
- fix(passthrough): skip `[DONE]` sentinels and non-JSON SSE frames in Anthropic streaming logging - [PR #30202](https://github.com/BerriAI/litellm/pull/30202)
- fix(cost): resolve `completion_cost` AttributeError on streaming Anthropic web_search responses, including the `server_tool_use` type-coercion prerequisite - [PR #27346](https://github.com/BerriAI/litellm/pull/27346)
- fix(proxy): authorize batch files using upload `target_model_names` - [PR #30009](https://github.com/BerriAI/litellm/pull/30009)
- fix(proxy): atomic merge for team model aliases and `team.models` on BYOK create - [PR #29528](https://github.com/BerriAI/litellm/pull/29528)
- feat(datadog): add team-scoped Datadog callback support - [PR #29947](https://github.com/BerriAI/litellm/pull/29947)
- feat(bedrock_mantle): add SigV4/IAM auth to the Responses API route - [PR #29788](https://github.com/BerriAI/litellm/pull/29788)
- fix(guardrails): read CrowdStrike AIDR identity from both metadata bags - [PR #29991](https://github.com/BerriAI/litellm/pull/29991)
- fix(slack): use the actual Slack return details for the `expires_in` default - [PR #29951](https://github.com/BerriAI/litellm/pull/29951)
- chore(deps): bump vitest, brace-expansion, pypdf, and tornado - [PR #30220](https://github.com/BerriAI/litellm/pull/30220)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.89.0...v1.89.1
