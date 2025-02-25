---
title: v1.55.10
slug: v1.55.10
date: 2024-12-24T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1743638400&v=beta&t=39KOXMUFedvukiWWVPHf3qI45fuQD7lNglICwN31DrI
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [batches, guardrails, team management, custom auth]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

# v1.55.10

`batches`, `guardrails`, `team management`, `custom auth`


<Image img={require('../../img/batches_cost_tracking.png')} />

<br/>

:::info

Get a free 7-day LiteLLM Enterprise trial here. [Start here](https://www.litellm.ai/#trial)

**No call needed**

:::

## ✨ Cost Tracking, Logging for Batches API (`/batches`)

Track cost, usage for Batch Creation Jobs. [Start here](https://docs.litellm.ai/docs/batches)

## ✨ `/guardrails/list` endpoint 

Show available guardrails to users. [Start here](https://litellm-api.up.railway.app/#/Guardrails)


## ✨ Allow teams to add models

This enables team admins to call their own finetuned models via litellm proxy. [Start here](https://docs.litellm.ai/docs/proxy/team_model_add)


## ✨ Common checks for custom auth

Calling the internal common_checks function in custom auth is now enforced as an enterprise feature. This allows admins to use litellm's default budget/auth checks within their custom auth implementation. [Start here](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)


## ✨ Assigning team admins

Team admins is graduating from beta and moving to our enterprise tier. This allows proxy admins to allow others to manage keys/models for their own teams (useful for projects in production). [Start here](https://docs.litellm.ai/docs/proxy/virtual_keys#restricting-key-generation)



