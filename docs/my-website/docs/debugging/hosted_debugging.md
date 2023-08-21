import Image from '@theme/IdealImage';

# Debugging Dashboard
LiteLLM offers a free hosted debugger UI for your api calls (https://admin.litellm.ai/). Useful if you're testing your LiteLLM server and need to see if the API calls were made successfully.

**Needs litellm>=0.1.438***

You can enable this setting `lite_debugger` as a callback. 

<Image img={require('../../img/dashboard.png')} alt="Dashboard" />

See our live dashboard 👉 [admin.litellm.ai](https://admin.litellm.ai/)

## Setup

By default, your dashboard is viewable at `admin.litellm.ai/<your_email>`. 

```
 ## Set your email
 os.environ["LITELLM_EMAIL"] = "your_user_email"
 
 ## LOG ON ALL 3 EVENTS
 litellm.input_callback = ["lite_debugger"]
 litellm.success_callback = ["lite_debugger"]
 litellm.failure_callback = ["lite_debugger"]

```

## Example Usage

```
 import litellm
 from litellm import embedding, completion
 import os 

 ## Set ENV variable 
 os.environ["LITELLM_EMAIL"] = "your_email"
 
 ## LOG ON ALL 3 EVENTS
 litellm.input_callback = ["lite_debugger"]
 litellm.success_callback = ["lite_debugger"]
 litellm.failure_callback = ["lite_debugger"]

 litellm.set_verbose = True

 user_message = "Hello, how are you?"
 messages = [{ "content": user_message,"role": "user"}]


 # openai call
 response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])

 # bad request call
 response = completion(model="chatgpt-test", messages=[{"role": "user", "content": "Hi 👋 - i'm a bad request"}])

```

