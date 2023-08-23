import Image from '@theme/IdealImage';

# Debugging Dashboard
LiteLLM offers a free UI to debug your calls + add new models at (https://admin.litellm.ai/). This is useful if you're testing your LiteLLM server and need to see if the API calls were made successfully **or** want to add new models without going into code. 

**Needs litellm>=0.1.438***

## Setup

Once created, your dashboard is viewable at - `admin.litellm.ai/<your_email>` [👋 Tell us if you need better privacy controls](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version?month=2023-08)

You can set your user email in 2 ways.  
- By setting it on the module - `litellm.email=<your_email>`.
- By setting it as an environment variable - `os.environ["LITELLM_EMAIL"] = "your_email"`.

<Image img={require('../../img/dashboard.png')} alt="Dashboard" />

See our live dashboard 👉 [admin.litellm.ai](https://admin.litellm.ai/)


## Example Usage

```
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

