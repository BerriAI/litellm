
import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Getting Started with UI Logs

View Spend, Token Usage, Key, Team Name for Each Request to LiteLLM


<Image img={require('../../img/ui_request_logs.png')}/>


## Overview

| Log Type | Tracked by Default |
|----------|-------------------|
| Success Logs | ✅ Yes |
| Error Logs | ✅ Yes |
| Request/Response Content Stored | ❌ No by Default, **opt in with `store_prompts_in_spend_logs`** |



**By default LiteLLM does not track the request and response content.**

## Tracking - Request / Response Content in Logs Page 

If you want to view request and response content on LiteLLM Logs, you need to opt in with this setting

```yaml
general_settings:
  store_prompts_in_spend_logs: true
```

<Image img={require('../../img/ui_request_logs_content.png')}/>


## Stop storing Error Logs in DB

If you do not want to store error logs in DB, you can opt out with this setting

```yaml
general_settings:
  disable_error_logs: True   # Only disable writing error logs to DB, regular spend logs will still be written unless `disable_spend_logs: True`
```

## Stop storing Spend Logs in DB

If you do not want to store spend logs in DB, you can opt out with this setting

```yaml
general_settings:
  disable_spend_logs: True   # Disable writing spend logs to DB
```

## Automatically Deleting Old Spend Logs

If you're storing spend logs, it might be a good idea to delete them regularly to keep the database fast.

LiteLLM lets you configure this in your `proxy_config.yaml`:

```yaml
general_settings:
  maximum_spend_logs_retention_period: "7d"  # Delete logs older than 7 days

  # Optional: how often to run cleanup
  maximum_spend_logs_retention_interval: "1d"  # Run once per day
```

You can control how many logs are deleted per run using this environment variable:

`SPEND_LOG_RUN_LOOPS=200  # Deletes up to 200,000 logs in one run`

Set `SPEND_LOG_CLEANUP_BATCH_SIZE` to control how many logs are deleted per batch (default `1000`).

For detailed architecture and how it works, see [Spend Logs Deletion](../proxy/spend_logs_deletion).


## What gets logged? 

[Here's a schema](https://github.com/BerriAI/litellm/blob/1cdd4065a645021aea931afb9494e7694b4ec64b/schema.prisma#L285) breakdown of what gets logged.
