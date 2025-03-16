
import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# UI Logs Page

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

