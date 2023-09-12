import Image from '@theme/IdealImage';
import QueryParamReader from '../../src/components/queryParamReader.js'

# [Beta] Monitor Logs in Production

:::note

This is in beta. Expect frequent updates, as we improve based on your feedback.

:::

LiteLLM provides an integration to let you monitor logs in production.

👉 Jump to our sample LiteLLM Dashboard: https://admin.litellm.ai/


<Image img={require('../../img/alt_dashboard.png')} alt="Dashboard" />

## Debug your first logs
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_OpenAI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### 1. Get your LiteLLM Token

Go to [admin.litellm.ai](https://admin.litellm.ai/) and copy the code snippet with your unique token

<Image img={require('../../img/hosted_debugger_usage_page.png')} alt="Usage" />

### 2. Set up your environment

**Add it to your .env**

```python
import os 

os.env["LITELLM_TOKEN"] = "e24c4c06-d027-4c30-9e78-18bc3a50aebb" # replace with your unique token

```

**Turn on LiteLLM Client**
```python
import litellm 
litellm.client = True
```

### 3. Make a normal `completion()` call
```python
import litellm 
from litellm import completion
import os 

# set env variables
os.environ["LITELLM_TOKEN"] = "e24c4c06-d027-4c30-9e78-18bc3a50aebb" # replace with your unique token
os.environ["OPENAI_API_KEY"] = "openai key"

litellm.use_client = True # enable logging dashboard 
messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)
```

Your `completion()` call print with a link to your session dashboard (https://admin.litellm.ai/<your_unique_token>)

In the above case it would be: [`admin.litellm.ai/e24c4c06-d027-4c30-9e78-18bc3a50aebb`](https://admin.litellm.ai/e24c4c06-d027-4c30-9e78-18bc3a50aebb)

Click on your personal dashboard link. Here's how you can find it 👇

<Image img={require('../../img/dash_output.png')} alt="Dashboard" />

[👋 Tell us if you need better privacy controls](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version?month=2023-08)  

### 3. Review request log 

Oh! Looks like our request was made successfully. Let's click on it and see exactly what got sent to the LLM provider. 

<Image img={require('../../img/dashboard_log_row.png')} alt="Dashboard Log Row" />    



Ah! So we can see that this request was made to a **Baseten** (see litellm_params > custom_llm_provider) for a model with ID - **7qQNLDB** (see model). The message sent was - `"Hey, how's it going?"` and the response received was - `"As an AI language model, I don't have feelings or emotions, but I can assist you with your queries. How can I assist you today?"`

<Image img={require('../../img/dashboard_log.png')} alt="Dashboard Log Row" />

:::info

🎉 Congratulations! You've successfully debugger your first log!

:::