import Image from '@theme/IdealImage';

# LiteLLM Client: Debug Logs + Instant LLM Deploys
LiteLLM offers a UI to:
* 1-Click Deploy LLMs - the client stores your api keys + model configurations
* Debug your Call Logs 

<Image img={require('../../img/dashboard.png')} alt="Dashboard" />

## Getting Started
All LiteLLM completion() calls auto-create a LiteLLM Client Dashboard
Example code snippet to create your dashboard:
```python
from litellm import completion
messages = [{ "content": "Hello, how are you?" ,"role": "user"}]
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

### Output with dashboard
All completion calls link to your session dashboard
```
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

## Usage
### Requirements
1. **Needs litellm>=0.1.438***
2. Set `litellm.email` account. You can set your user email in 2 ways.  
    - By setting it on the module - `litellm.email=<your_email>`.
    - By setting it as an environment variable - `os.environ["LITELLM_EMAIL"] = "your_email"`.

## Code Setup
```python
import litellm

## Setup for activating / using the litellm dashboard
litellm.email = "test_email@test.com"

```

## Using LiteLLM Dashboard - Add New LLMs
LiteLLM allows you to add a new model using the liteLLM Dashboard 
Go to `admin.litellm.ai/<your_email>`
Navigate to the 'Add New LLM' Section
<Image img={require('../../img/add_model.png')} alt="Dashboard" />

- Select Provider
- Select your LLM 
- Add your LLM Key

### Using your new LLM - Completion() with the LiteLLM Dashboard
```python
from litellm import embedding, completion
# keys set in admin.litellm.ai/<your_email> or .env OPENAI_API_KEY
messages = [{ "content": "Hello, how are you?" ,"role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

## Using LiteLLM Dashboard - Debug your Call Logs 

```python
from litellm import embedding, completion
# keys set in admin.litellm.ai/<your_email> or .env OPENAI_API_KEY
messages = [{ "content": "Hello, how are you?" ,"role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

See your Logs below