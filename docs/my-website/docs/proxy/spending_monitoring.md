# Using at Scale (1M+ rows in DB)

This document is a guide for using LiteLLM Proxy once you have crossed 1M+ rows in the LiteLLM Spend Logs Database.

<iframe width="840" height="500" src="https://www.loom.com/embed/eafd90d5374d4633b99c441fb04df351" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Why is UI Usage Tracking disabled?
- Heavy database queries on `LiteLLM_Spend_Logs` (once it has 1M+ rows) can slow down your LLM API requests. **We do not want this happening**

## Solutions for Usage Tracking

Step 1. **Export Logs to Cloud Storage**
   - [Send logs to S3, GCS, or Azure Blob Storage](https://docs.litellm.ai/docs/proxy/logging)
   - [Log format specification](https://docs.litellm.ai/docs/proxy/logging_spec)

Step 2. **Analyze Data**
   - Use tools like [Redash](https://redash.io/), [Databricks](https://www.databricks.com/), [Snowflake](https://www.snowflake.com/en/) to analyze exported logs

[Optional] Step 3. **Disable Spend + Error Logs to LiteLLM DB**

[See Instructions Here](./prod#6-disable-spend_logs--error_logs-if-not-using-the-litellm-ui)

Disabling this will prevent your LiteLLM DB from growing in size, which will help with performance (prevent health checks from failing).

## Need an Integration? Get in Touch

- Request a logging integration on [Github Issues](https://github.com/BerriAI/litellm/issues)
- Get in [touch with LiteLLM Founders](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)
- Get a 7-day free trial of LiteLLM [here](https://litellm.ai#trial)



