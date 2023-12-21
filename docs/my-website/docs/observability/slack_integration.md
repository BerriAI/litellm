import Image from '@theme/IdealImage';

# Slack - Logging LLM Input/Output, Exceptions

<Image img={require('../../img/slack.png')} />

:::info
We want to learn how we can make the callbacks better! Meet the LiteLLM [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 

## Pre-Requisites

### Step 1
```shell
pip install litellm
```

### Step 2
Get a slack webhook url from https://api.slack.com/messaging/webhooks



## Quick Start
### Create a custom Callback to log to slack
We create a custom callback, to log to slack webhooks, see [custom callbacks on litellm](https://docs.litellm.ai/docs/observability/custom_callback)
```python
def send_slack_alert(
        kwargs,
        completion_response,
        start_time,
        end_time,
):
    print(
        "in custom slack callback func"
    )
    import requests
    import json

    # Define the Slack webhook URL
    # get it from https://api.slack.com/messaging/webhooks
    slack_webhook_url = os.environ['SLACK_WEBHOOK_URL']   # "https://hooks.slack.com/services/<>/<>/<>"

    # Remove api_key from kwargs under litellm_params
    if kwargs.get('litellm_params'):
        kwargs['litellm_params'].pop('api_key', None)
        if kwargs['litellm_params'].get('metadata'):
            kwargs['litellm_params']['metadata'].pop('deployment', None)
    # Remove deployment under metadata
    if kwargs.get('metadata'):
        kwargs['metadata'].pop('deployment', None)
    # Prevent api_key from being logged
    if kwargs.get('api_key'):
        kwargs.pop('api_key', None)

    # Define the text payload, send data available in litellm custom_callbacks
    text_payload = f"""LiteLLM Logging: kwargs: {str(kwargs)}\n\n, response: {str(completion_response)}\n\n, start time{str(start_time)} end time: {str(end_time)}
    """
    payload = {
        "text": text_payload
    }

    # Set the headers
    headers = {
        "Content-type": "application/json"
    }

    # Make the POST request
    response = requests.post(slack_webhook_url, json=payload, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print("Message sent successfully to Slack!")
    else:
        print(f"Failed to send message to Slack. Status code: {response.status_code}")
        print(response.json())
```

### Pass callback to LiteLLM
```python
litellm.success_callback = [send_slack_alert]
```

```python
import litellm
litellm.success_callback = [send_slack_alert] # log success
litellm.failure_callback = [send_slack_alert] # log exceptions

# this will raise an exception
response = litellm.completion(
    model="gpt-2",
    messages=[
        {
            "role": "user",
            "content": "Hi 👋 - i'm openai"
        }
    ]
)
```
## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
