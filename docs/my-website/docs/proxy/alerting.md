# 🚨 Alerting / Webhooks

Get alerts for:

- Hanging LLM api calls
- Slow LLM api calls
- Failed LLM api calls
- Budget Tracking per key/user
- Spend Reports - Weekly & Monthly spend per Team, Tag
- Failed db read/writes
- Model outage alerting
- Daily Reports:
    - **LLM** Top 5 slowest deployments
    - **LLM** Top 5 deployments with most failed requests
- **Spend** Weekly & Monthly spend per Team, Tag


## Quick Start

Set up a slack alert channel to receive alerts from proxy.

### Step 1: Add a Slack Webhook URL to env

Get a slack webhook url from https://api.slack.com/messaging/webhooks

You can also use Discord Webhooks, see [here](#using-discord-webhooks)

### Step 2: Update config.yaml 

- Set `SLACK_WEBHOOK_URL` in your proxy env to enable Slack alerts.
- Just for testing purposes, let's save a bad key to our proxy.

```yaml
model_list: 
    model_name: "azure-model"
    litellm_params:
        model: "azure/gpt-35-turbo"
        api_key: "my-bad-key" # 👈 bad key

general_settings: 
    alerting: ["slack"]
    alerting_threshold: 300 # sends alerts if requests hang for 5min+ and responses take 5min+ 

environment_variables:
    SLACK_WEBHOOK_URL: "https://hooks.slack.com/services/<>/<>/<>"
    SLACK_DAILY_REPORT_FREQUENCY: "86400"  # 24 hours; Optional: defaults to 12 hours
```


### Step 3: Start proxy

```bash
$ litellm --config /path/to/config.yaml
```

## Testing Alerting is Setup Correctly

Make a GET request to `/health/services`, expect to see a test slack alert in your provided webhook slack channel

```shell
curl -X GET 'http://localhost:4000/health/services?service=slack' \
  -H 'Authorization: Bearer sk-1234'
```

## Advanced - Redacting Messages from Alerts

By default alerts show the `messages/input` passed to the LLM. If you want to redact this from slack alerting set the following setting on your config


```shell
general_settings:
  alerting: ["slack"]
  alert_types: ["spend_reports"] 

litellm_settings:
  redact_messages_in_exceptions: True
```




## Advanced - Opting into specific alert types

Set `alert_types` if you want to Opt into only specific alert types

```shell
general_settings:
  alerting: ["slack"]
  alert_types: ["spend_reports"] 
```

All Possible Alert Types

```python
AlertType = Literal[
    "llm_exceptions",
    "llm_too_slow",
    "llm_requests_hanging",
    "budget_alerts",
    "db_exceptions",
    "daily_reports",
    "spend_reports",
    "cooldown_deployment",
    "new_model_added",
    "outage_alerts",
]

```


## Advanced - Using Discord Webhooks

Discord provides a slack compatible webhook url that you can use for alerting

##### Quick Start

1. Get a webhook url for your discord channel 

2. Append `/slack` to your discord webhook - it should look like

```
"https://discord.com/api/webhooks/1240030362193760286/cTLWt5ATn1gKmcy_982rl5xmYHsrM1IWJdmCL1AyOmU9JdQXazrp8L1_PYgUtgxj8x4f/slack"
```

3. Add it to your litellm config 

```yaml
model_list: 
    model_name: "azure-model"
    litellm_params:
        model: "azure/gpt-35-turbo"
        api_key: "my-bad-key" # 👈 bad key

general_settings: 
    alerting: ["slack"]
    alerting_threshold: 300 # sends alerts if requests hang for 5min+ and responses take 5min+ 

environment_variables:
    SLACK_WEBHOOK_URL: "https://discord.com/api/webhooks/1240030362193760286/cTLWt5ATn1gKmcy_982rl5xmYHsrM1IWJdmCL1AyOmU9JdQXazrp8L1_PYgUtgxj8x4f/slack"
```

That's it ! You're ready to go !

## Advanced - [BETA] Webhooks for Budget Alerts

**Note**: This is a beta feature, so the spec might change.

Set a webhook to get notified for budget alerts. 

1. Setup config.yaml

Add url to your environment, for testing you can use a link from [here](https://webhook.site/)

```bash
export WEBHOOK_URL="https://webhook.site/6ab090e8-c55f-4a23-b075-3209f5c57906"
```

Add 'webhook' to config.yaml
```yaml
general_settings: 
  alerting: ["webhook"] # 👈 KEY CHANGE
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```bash
curl -X GET --location 'http://0.0.0.0:4000/health/services?service=webhook' \
--header 'Authorization: Bearer sk-1234'
```

**Expected Response**

```bash
{
  "spend": 1, # the spend for the 'event_group'
  "max_budget": 0, # the 'max_budget' set for the 'event_group'
  "token": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
  "user_id": "default_user_id",
  "team_id": null,
  "user_email": null,
  "key_alias": null,
  "projected_exceeded_data": null,
  "projected_spend": null,
  "event": "budget_crossed", # Literal["budget_crossed", "threshold_crossed", "projected_limit_exceeded"]
  "event_group": "user",
  "event_message": "User Budget: Budget Crossed"
}
```

## **API Spec for Webhook Event**

- `spend` *float*: The current spend amount for the 'event_group'.
- `max_budget` *float or null*: The maximum allowed budget for the 'event_group'. null if not set. 
- `token` *str*: A hashed value of the key, used for authentication or identification purposes.
- `customer_id` *str or null*: The ID of the customer associated with the event (optional).
- `internal_user_id` *str or null*: The ID of the internal user associated with the event (optional).
- `team_id` *str or null*: The ID of the team associated with the event (optional).
- `user_email` *str or null*: The email of the internal user associated with the event (optional).
- `key_alias` *str or null*: An alias for the key associated with the event (optional).
- `projected_exceeded_date` *str or null*: The date when the budget is projected to be exceeded, returned when 'soft_budget' is set for key (optional).
- `projected_spend` *float or null*: The projected spend amount, returned when 'soft_budget' is set for key (optional).
- `event` *Literal["budget_crossed", "threshold_crossed", "projected_limit_exceeded"]*: The type of event that triggered the webhook. Possible values are:
    * "spend_tracked": Emitted whenver spend is tracked for a customer id. 
    * "budget_crossed": Indicates that the spend has exceeded the max budget.
    * "threshold_crossed": Indicates that spend has crossed a threshold (currently sent when 85% and 95% of budget is reached).
    * "projected_limit_exceeded": For "key" only - Indicates that the projected spend is expected to exceed the soft budget threshold.
- `event_group` *Literal["customer", "internal_user", "key", "team", "proxy"]*: The group associated with the event. Possible values are:
    * "customer": The event is related to a specific customer
    * "internal_user": The event is related to a specific internal user.
    * "key": The event is related to a specific key.
    * "team": The event is related to a team.
    * "proxy": The event is related to a proxy.

- `event_message` *str*: A human-readable description of the event.

## Advanced - Region-outage alerting (✨ Enterprise feature)

:::info
[Get a free 2-week license](https://forms.gle/P518LXsAZ7PhXpDn8)
:::

Setup alerts if a provider region is having an outage. 

```yaml
general_settings:
    alerting: ["slack"]
    alert_types: ["region_outage_alerts"] 
```

By default this will trigger if multiple models in a region fail 5+ requests in 1 minute. '400' status code errors are not counted (i.e. BadRequestErrors).

Control thresholds with: 

```yaml
general_settings:
    alerting: ["slack"]
    alert_types: ["region_outage_alerts"] 
    alerting_args:
        region_outage_alert_ttl: 60 # time-window in seconds
        minor_outage_alert_threshold: 5 # number of errors to trigger a minor alert
        major_outage_alert_threshold: 10 # number of errors to trigger a major alert
```