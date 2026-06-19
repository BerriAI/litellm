---
slug: managing-ai-spend
title: "The CFO's Guide to Managing AI Spend with LiteLLM"
date: 2026-06-19T08:00:00
authors:
  - ishaan-alt
  - krrish
description: "How engineering leaders and CFOs use LiteLLM's AI Gateway to track, budget, and control LLM spend across teams, models, and projects — with real product walkthroughs."
tags: [spend tracking, budgets, AI gateway, enterprise, cost management]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

![LiteLLM AI Spend Management](/img/litellm_managing_ai_spend.png)

Your company started with one team calling GPT-4. Now you have 15 teams, 6 model providers, hundreds of API keys, and a monthly bill that surprises everyone — including Finance.

This is the reality for most companies scaling AI in production. The models are easy to plug in. Knowing who spent what, setting limits before the bill arrives, and giving your CFO a dashboard they can actually read — that's the hard part.

LiteLLM solves this today. Not as a beta. Not behind a waitlist. As a self-hosted, open source AI Gateway that 45,000+ developers already run in production.

{/* truncate */}

## The problem: AI spend is invisible until the invoice arrives

Most companies discover their AI costs the same way they discover cloud costs went sideways — at month-end, staring at a provider invoice with no breakdown by team, project, or use case.

Here's what typically goes wrong:

1. **No central visibility.** Teams call OpenAI, Anthropic, and Bedrock directly. Each has its own billing. Nobody sees the full picture until someone manually aggregates invoices.
2. **No per-team attribution.** Even when one proxy exists, spend is tracked at the org level. When the VP of Engineering asks "how much is the search team spending on Claude?", nobody knows.
3. **No preventive controls.** Budgets are set retroactively. A runaway agent racks up $50K over a weekend. The alert comes on Monday.
4. **No model-level optimization.** You're paying for GPT-4 on tasks where GPT-4o Mini would return the same quality. But without per-model spend data, there's nothing to optimize.

## How LiteLLM gives you full spend visibility

LiteLLM sits between your applications and every LLM provider. Every API call flows through the gateway, and every call gets tracked — automatically, with zero code changes beyond pointing your `base_url` to LiteLLM.

### Real-time spend dashboard

The Usage page gives you an instant view of total spend, broken down by time period, API key, and model:

<Image img={require('../../img/ui_usage_v189.png')} />

*LiteLLM v1.89 Usage Dashboard — Usage Metrics (total requests, successful/failed, average cost per request, total tokens), Daily Spend chart, Top Virtual Keys, Top Models, and Spend by Provider. Filter by user, toggle between Cost, Model Activity, Key Activity, and Endpoint Activity views.*

This is the view your CFO needs. Total spend at the top. Trends over time. Which keys are driving cost. Which models are eating the budget. All in one screen, updating in real time.

For teams already running in production, the dashboard shows real spend data across all your API keys and models:

<Image img={require('../../img/admin_ui_spend.png')} />

*Production spend view — monthly spend trends, top API keys by cost, and top models by usage.*

### Team-based spend tracking

For organizations with multiple teams, LiteLLM tracks spend per team automatically. Create teams, assign API keys, and the dashboard breaks down every dollar:

<Image img={require('../../img/release_notes/new_team_usage.png')} />

*Team Usage view — filter by one or multiple teams, see total requests, tokens, and spend. Drill into individual keys within each team.*

Your platform team can see their $372K spend. Your product team can see their $241K. No manual tagging required — it's structural, built into how keys are issued.

### Tag-based cost attribution

Need to track spend by project, environment, or business unit — not just by team? Tags give you flexible cost attribution across any dimension:

<Image img={require('../../img/release_notes/tag_management.png')} />

*Tag Management — create tags like "private" or "general-use-case", restrict which models each tag can access, and set per-tag budgets.*

Tags work two ways:
- **Attach to keys** — every request from that key inherits the tag automatically
- **Pass per request** — add `metadata.tags` or the `x-litellm-tags` header for request-level attribution

Either way, your `/spend/tags` endpoint gives you spend broken down by any tag you define.

### Customer-level usage analytics

If you're building AI-powered products and need to track spend per customer (for billing, cost allocation, or margin analysis), LiteLLM has a dedicated Customer Usage view:

<Image img={require('../../img/customer_usage.png')} />

*Customer Usage — total spend, requests, tokens, and daily trends per customer. Export data for finance and billing systems.*

This is how companies building on top of LLMs understand their unit economics. Pass a `user` parameter on each request, and LiteLLM attributes spend to that customer automatically.

## Setting budgets before the bill arrives

Tracking spend is step one. Preventing overruns is step two. LiteLLM gives you budget controls at every level of your organization.

### Global budgets

Cap total spend across your entire proxy:

```yaml
litellm_settings:
  max_budget: 100000    # $100K USD
  budget_duration: 30d  # resets monthly
```

### Per-team budgets

Give each team their own spend ceiling:

```bash
curl -X POST 'http://localhost:4000/team/new' \
  -H 'Authorization: Bearer sk-master-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_alias": "platform-team",
    "max_budget": 50000,
    "budget_duration": "30d"
  }'
```

When a team hits their budget, requests are rejected with a clear error. No surprise bills. No weekend runaway costs.

### Per-key budgets with multiple windows

The most granular control: set budgets on individual API keys, with multiple concurrent time windows. Cap a key at $100/day AND $2,000/month:

```bash
curl 'http://localhost:4000/key/generate' \
  -H 'Authorization: Bearer sk-master-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "platform-team",
    "budget_limits": [
      {"budget_duration": "24h", "max_budget": 100},
      {"budget_duration": "30d", "max_budget": 2000}
    ]
  }'
```

Each window tracks and resets independently. Daily limits catch runaway agents. Monthly limits keep Finance happy.

### Budgets UI

The Budgets page lets you create and manage budgets directly from the UI — set max budget, TPM, and RPM limits:

<Image img={require('../../img/ui_budgets_v189.png')} />

*LiteLLM v1.89 Budgets page — create budgets with max spend, TPM, and RPM limits to assign to teams or customers.*

### Tag-based budgets

Budget by business unit, not just by team structure:

<Image img={require('../../img/tag_budget1.png')} />

*Create a tag with a max budget — when spend on that tag exceeds the limit, requests tagged with it are rejected.*

### Cost Tracking Settings

Fine-tune your cost calculations with provider discounts, fee/price margins, and a built-in pricing calculator:

<Image img={require('../../img/ui_cost_tracking_v189.png')} />

*Cost Tracking Settings — apply provider discounts, add margins for internal billing, and estimate costs with the pricing calculator.*

### Per-request cost tracking

Every single LLM call returns its cost in the response headers:

<Image img={require('../../img/response_cost_img.png')} />

*Per-request cost tracking — see exact cost, tokens (prompt + completion), cache hits, model, provider, and duration for every call.*

This means your application can track costs programmatically. Build internal dashboards. Alert on anomalies. Charge back to customers. The data is there on every response.

## Provider budget routing: optimize across clouds

Running multiple providers? LiteLLM can route requests based on provider budgets. Set $50K/month for OpenAI and $30K/month for Anthropic — when one provider's budget is exhausted, traffic automatically shifts:

```yaml
litellm_settings:
  provider_budget_config:
    openai:
      budget_limit: 50000
      time_period: "30d"
    anthropic:
      budget_limit: 30000
      time_period: "30d"
```

Monitor remaining budgets in real time via `/provider/budgets`, or track them with Prometheus metrics for your existing monitoring stack.

## Spend reports for Finance

Need to generate reports for your finance team? The `/global/spend/report` endpoint returns spend broken down by team, customer, API key, or internal user:

```bash
# Spend by team for Q2
curl -X GET 'http://localhost:4000/global/spend/report?\
  start_date=2026-04-01&\
  end_date=2026-06-30&\
  group_by=team' \
  -H 'Authorization: Bearer sk-master-key'
```

```bash
# Daily activity by user
curl -X GET 'http://localhost:4000/user/daily/activity?\
  start_date=2026-06-01&\
  end_date=2026-06-19' \
  -H 'Authorization: Bearer sk-user-key'
```

Export data from the UI, pull from the API, or pipe directly to your BI tools. The data is structured and queryable.

## Why this matters now

AI spend is the new cloud spend. The same patterns that led to $100B+ in cloud waste are repeating with LLMs:

- Teams spin up experiments with frontier models and never switch to cheaper alternatives
- Agents run unsupervised with no spend caps
- Multiple teams buy capacity from the same provider separately
- Nobody knows the true cost-per-feature until the quarterly review

The companies that get ahead of this are the ones treating AI spend as a first-class operational concern — not an afterthought.

LiteLLM gives you the same controls you expect from your cloud infrastructure: real-time visibility, per-team budgets, alerting, and programmatic access to every dollar spent. Across 100+ LLM providers. Self-hosted in your environment. Open source.

## Get started

LiteLLM is open source and free to self-host:

```bash
docker pull ghcr.io/berriai/litellm:main-latest
```

- **Docs:** [Spend Tracking](https://docs.litellm.ai/docs/proxy/cost_tracking)
- **Docs:** [Budgets & Rate Limits](https://docs.litellm.ai/docs/proxy/users)
- **Docs:** [Tag-Based Budgets](https://docs.litellm.ai/docs/proxy/tag_budgets)
- **GitHub:** [BerriAI/litellm](https://github.com/BerriAI/litellm) — 45K+ stars
- **Enterprise:** [Request a trial](https://litellm.ai) for SSO, audit logs, and advanced governance

Every feature shown in this post works today. No waitlist. No beta. Ship it this week.
