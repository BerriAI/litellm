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



