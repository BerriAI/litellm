---
title: v1.59.0
slug: v1.59.0
date: 2025-01-17T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [admin ui, logging, db schema]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

# v1.59.0



:::info

Get a 7 day free trial for LiteLLM Enterprise [here](https://litellm.ai/#trial).

**no call needed**

:::

## UI Improvements

### [Opt In] Admin UI - view messages / responses 

You can now view messages and response logs on Admin UI.

<Image img={require('../../img/release_notes/ui_logs.png')} />

How to enable it - add `store_prompts_in_spend_logs: true` to your `proxy_config.yaml`

Once this flag is enabled, your `messages` and `responses` will be stored in the `LiteLLM_Spend_Logs` table.

```yaml
general_settings:
  store_prompts_in_spend_logs: true
```

## DB Schema Change

Added `messages` and `responses` to the `LiteLLM_Spend_Logs` table.

**By default this is not logged.** If you want `messages` and `responses` to be logged, you need to opt in with this setting 

```yaml
general_settings:
  store_prompts_in_spend_logs: true
```


