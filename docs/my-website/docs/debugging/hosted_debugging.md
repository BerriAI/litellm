import Image from '@theme/IdealImage';

# LiteLLM Client: 1-Click Deploy LLMs + Debug Logs
LiteLLM offers a UI to:
* 1-Click Deploy LLMs - the client stores your api keys + model configurations
* Debug your Call Logs 

<Image img={require('../../img/dashboard.png')} alt="Dashboard" />

## Getting Started
<a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/drive/1y2ChqxJOwEByThibxYMEpY5P6_RTNjj4?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
* Make a `litellm.completion()` call 👉 get your debugging dashboard  

Example Code: Regular `litellm.completion()` call:
```python
from litellm import completion
messages = [{ "content": "Hello, how are you?" ,"role": "user"}]
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

## Completion() Output with dashboard
All `completion()` calls print with a link to your session dashboard

<Image img={require('../../img/dash_output.png')} alt="Dashboard" />


Example Output from litellm completion
```bash
Here's your LiteLLM Dashboard 👉 https://admin.litellm.ai/88911906-d786-44f2-87c7-9720e6031b45
<OpenAIObject chat.completion id=chatcmpl-7r6LtlUXYYu0QayfhS3S0OzroiCel at 0x7fb307375030> JSON: {
  "id": "chatcmpl-7r6LtlUXYYu0QayfhS3S0OzroiCel",
  "object": "chat.completion",
  "created": 1692890157,
  "model": "gpt-3.5-turbo-0613",
..............

```

Once created, your dashboard is viewable at - `admin.litellm.ai/<your_email>` [👋 Tell us if you need better privacy controls](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version?month=2023-08)
See our live dashboard 👉 [admin.litellm.ai](https://admin.litellm.ai/)



## Code Setup
```python
import litellm

## Setup for activating / using the litellm dashboard
litellm.email = "test_email@test.com"

```

## LiteLLM Dashboard - 1-Click Deploy LLMs
LiteLLM allows you to add a new model using the liteLLM Dashboard 

Navigate to the 'Add New LLM' Section
<Image img={require('../../img/add_model.png')} alt="Dashboard" />
- Select Provider
- Select your LLM 
- Add your LLM Key

## LiteLLM Dashboard - Debug Logs 
All your `completion()` and `embedding()` call logs are available on `admin.litellm.ai/<your-token>`
See your Logs below

### Using your new LLM - Completion() with the LiteLLM Dashboard
```python
from litellm import embedding, completion
# keys set in admin.litellm.ai/<your_email> or .env OPENAI_API_KEY
messages = [{ "content": "Hello, how are you?" ,"role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

