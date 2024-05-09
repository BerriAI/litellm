# 🚨 Alerting 

Get alerts for:
- Hanging LLM api calls
- Failed LLM api calls
- Slow LLM api calls
- Budget Tracking per key/user:
    - When a User/Key crosses their Budget 
    - When a User/Key is 15% away from crossing their Budget
- Failed db read/writes

## Quick Start

Set up a slack alert channel to receive alerts from proxy.

### Step 1: Add a Slack Webhook URL to env

Get a slack webhook url from https://api.slack.com/messaging/webhooks


### Step 2: Update config.yaml 

Let's save a bad key to our proxy

```yaml
model_list: 
    model_name: "azure-model"
    litellm_params:
        model: "azure/gpt-35-turbo"
        api_key: "my-bad-key" # 👈 bad key

general_settings: 
    alerting: ["slack"]
    alerting_threshold: 300 # sends alerts if requests hang for 5min+ and responses take 5min+ 

```

Set `SLACK_WEBHOOK_URL` in your proxy env

```shell
SLACK_WEBHOOK_URL: "https://hooks.slack.com/services/<>/<>/<>"
```

### Step 3: Start proxy

```bash
$ litellm --config /path/to/config.yaml
```