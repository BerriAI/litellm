import Image from '@theme/IdealImage';

# 🚨 Budget Alerting

**Alerts when a project will exceed it’s planned limit**

<Image img={require('../../img/budget_alerts.png')} />

## Quick Start

### 1. Setup Slack Alerting on your Proxy Config.yaml 

**Add Slack Webhook to your env**
Get a slack webhook url from https://api.slack.com/messaging/webhooks


Set `SLACK_WEBHOOK_URL` in your proxy env

```shell
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/<>/<>/<>"
```

**Update proxy config.yaml with slack alerting**  

Add `general_settings:alerting`
```yaml
model_list: 
    model_name: "azure-model"
    litellm_params:
        model: "azure/gpt-35-turbo"

general_settings: 
    alerting: ["slack"]
```



Start proxy
```bash
$ litellm --config /path/to/config.yaml
```


### 2. Create API Key on Proxy Admin UI
The Admin UI is found on `your-litellm-proxy-endpoint/ui`, example `http://localhost:4000/ui/` 

- Set a key name 
- Set a Soft Budget on when to get alerted 

<Image img={require('../../img/create_key.png')} />


### 3. Test Slack Alerting on Admin UI
After creating a key on the Admin UI, click on "Test Slack Alert" to send a test alert to your Slack channel
<Image img={require('../../img/test_alert.png')} />

### 4. Check Slack 

When the test alert works, you should expect to see this on your alerts slack channel 

<Image img={require('../../img/budget_alerts.png')} />