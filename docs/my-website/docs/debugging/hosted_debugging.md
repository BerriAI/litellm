import Image from '@theme/IdealImage';

# LiteLLM Client - Debugging Dashboard
LiteLLM offers a UI to:
* Add New LLMs - Store your API Keys, Model Configurations
* Debug your Call Logs 

Once created, your dashboard is viewable at - `admin.litellm.ai/<your_email>` [👋 Tell us if you need better privacy controls](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version?month=2023-08)
See our live dashboard 👉 [admin.litellm.ai](https://admin.litellm.ai/)

<Image img={require('../../img/dashboard.png')} alt="Dashboard" />

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