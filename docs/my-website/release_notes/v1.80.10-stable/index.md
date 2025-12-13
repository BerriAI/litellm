---
title: "[Preview] v1.80.10.rc.1"
slug: "v1-80-10"
date: 2025-12-13T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.80.10.rc.1
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.80.10
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Agent (A2A) Usage UI** - [Track and visualize agent (A2A) spend directly in the dashboard](../../docs/proxy/agent_usage)


---

### Agent (A2A) Usage UI

<Image
img={require('../../img/agent_usage.png')}
style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

Users can now filter usage statistics by agents, providing the same granular filtering capabilities available for teams, organizations, and customers.

**Details:**

- Filter usage analytics, spend logs, and activity metrics by agent ID
- View breakdowns on a per-agent basis
- Consistent filtering experience across all usage and analytics views

---

## New Providers and Endpoints

### New Providers (5 new providers)

| Provider | Supported LiteLLM Endpoints | Description |
| -------- | ------------------- | ----------- |

### New LLM API Endpoints (2 new endpoints)

| Endpoint | Method | Description | Documentation |
| -------- | ------ | ----------- | ------------- |


---

## New Models / Updated Models

#### New Model Support (33 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |


#### Features


### Bug Fixes


---

## LLM API Endpoints

#### Features

#### Bugs

---

## Management Endpoints / UI

#### Features

#### Bugs

---

## AI Integrations (2 new integrations)



---

## Spend Tracking, Budgets and Rate Limiting



---

## MCP Gateway

---

## Agent Gateway (A2A)


---

## Performance / Loadbalancing / Reliability improvements


---

## Documentation Updates


---

## Infrastructure / CI/CD


---

## New Contributors


---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.80.8...v1.80.10)**

