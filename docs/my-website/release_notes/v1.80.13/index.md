---
title: "[Preview] v1.80.13 - Google Interactions API"
slug: "v1-80-13"
date: 2026-01-10T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.80.13.rc.1
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.80.13
```

</TabItem>
</Tabs>

---

## Key Highlights

- **UI Usage - Endpoint Activity** - Users can now see Endpoint Activity Metrics in the UI.


---

### UI Usage - Endpoint Activity

<Image
img={require('../../img/ui_endpoint_activity.png')}
style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

Users can now see Endpoint Activity Metrics in the UI.

---

## New Providers and Endpoints

### New Providers (X new providers)

| Provider | Supported LiteLLM Endpoints | Description |
| -------- | ------------------- | ----------- |


### New LLM API Endpoints (X new endpoints)

| Endpoint | Method | Description | Documentation |
| -------- | ------ | ----------- | ------------- |


---

## New Models / Updated Models

#### New Model Support (X new models)

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

## AI Integrations

### Logging


### Guardrails



### Secret Managers



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

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.80.11...v1.80.13.rc.1)**

