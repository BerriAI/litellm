# 🚨 Alerting 

Get alerts for:

- Hanging LLM api calls
- Failed LLM api calls
- Slow LLM api calls
- Budget Tracking per key/user:
  - When a User/Key crosses their Budget
  - When a User/Key is 15% away from crossing their Budget
- Failed db read/writes

As a bonus, you can also get "daily reports" posted to your slack channel.
These reports contain key metrics like:

- Top 5 deployments with most failed requests
- Top 5 slowest deployments

## Quick Start

Set up a slack alert channel to receive alerts from proxy.

### Step 1: Add a Slack Webhook URL to env

Get a slack webhook url from https://api.slack.com/messaging/webhooks


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