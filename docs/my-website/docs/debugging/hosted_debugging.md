import Image from '@theme/IdealImage';

# LiteLLM Client - Debugging Dashboard
LiteLLM offers a UI to
* Add New LLMs - Store your API Keys, Model Configurations
* Debug your Call Logs 
<Image img={require('../../img/dashboard.png')} alt="Dashboard" />

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
 from litellm import embedding, completion
 
 ## Set your email
 litellm.email = "test_email@test.com"

 user_message = "Hello, how are you?"
 messages = [{ "content": user_message,"role": "user"}]


 # openai call
 response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])

 # bad request call
 response = completion(model="chatgpt-test", messages=[{"role": "user", "content": "Hi 👋 - i'm a bad request"}])
```

