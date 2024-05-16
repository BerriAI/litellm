# 🚨 Alerting 

Get alerts for:

- Hanging LLM api calls
- Slow LLM api calls
- Failed LLM api calls
- Budget Tracking per key/user
- Spend Reports - Weekly & Monthly spend per Team, Tag
- Failed db read/writes
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


## Extras

### Using Discord Webhooks

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
