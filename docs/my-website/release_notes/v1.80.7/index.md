---
title: "[PREVIEW] v1.80.5.rc.2 - Gemini 3.0 Support"
slug: "v1-80-5"
date: 2025-11-22T10:00:00
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

```showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.80.5.rc.2
```

</TabItem>

<TabItem value="pip" label="Pip">

```showLineNumbers title="pip install litellm"
pip install litellm==1.80.5
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Gemini 3** - [Day-0 support for Gemini 3 models with thought signatures](../../blog/gemini_3)

---

### Organization Usage

<Image
img={require('../../img/release_notes/organization_usage.png')}
style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

Users can now filter usage statistics by organization, providing the same granular filtering capabilities available for teams.

**Details:**

- Filter usage analytics, spend logs, and activity metrics by organization ID
- View organization-level breakdowns alongside existing team and user-level filters
- Consistent filtering experience across all usage and analytics views

---

### New Providers

| Provider                                                            | Supported Endpoints    | Description                         |
| ------------------------------------------------------------------- | ---------------------- | ----------------------------------- |
| **[Docker Model Runner](../../docs/providers/docker_model_runner)** | `/v1/chat/completions` | Run LLM models in Docker containers |

---

## New Models / Updated Models

#### New Model Support

| Provider | Model           | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features                                    |
| -------- | --------------- | -------------- | ------------------- | -------------------- | ------------------------------------------- |
| Azure    | `azure/gpt-5.1` | 272K           | $1.38               | $11.00               | Reasoning, vision, PDF input, responses API |

#### Features

### Bug Fixes

- **[OpenAI](../../docs/providers/openai)**

- **General**

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**

#### Bugs

- **General**

---

## Management Endpoints / UI

#### Features

#### Bugs

---

## AI Integrations

### Logging

---

## Performance / Loadbalancing / Reliability improvements

---

## Documentation Updates

- **Provider Documentation**

- **API Documentation**

- **General Documentation**

---

## Infrastructure / CI/CD

- **UI Testing**

- **Dependency Management**

- **Migration**

- **Config**

- **Release Notes**

- **Investigation**

---

## New Contributors

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.80.0-nightly...v1.80.5.rc.2)**
