import Image from '@theme/IdealImage';

# LiteLLM Client: 1-Click Deploy LLMs + Debug Logs
LiteLLM offers a UI to:
* 1-Click Deploy LLMs - the client stores your api keys + model configurations
* Debug your Call Logs 

👉 Jump to our sample LiteLLM Dashboard: https://admin.litellm.ai/krrish@berri.ai

<Image img={require('../../img/dashboard.png')} alt="Dashboard" />

## Getting Started
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_OpenAI.ipynb">
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

## Opt-Out of using LiteLLM Client
If you want to opt out of using LiteLLM client you can set
```python
litellm.use_client = True
```
## Persisting your dashboard
If you want to use the same dashboard for your project set
`litellm.token` in code or your .env as `LITELLM_TOKEN`
All generated dashboards come with a token
```python
import litellm
litellm.token = "80888ede-4881-4876-ab3f-765d47282e66"
```

## LiteLLM Dashboard - 1-Click Deploy LLMs
LiteLLM allows you to add a new model using the liteLLM Dashboard 

Navigate to the 'Add New LLM' Section:
* Select Provider
* Select your LLM 
* Add your LLM Key

<Image img={require('../../img/add_model.png')} alt="Dashboard" />

After adding your new LLM, LiteLLM securely stores your API key and model configs. 
## Using `completion() with LiteLLM Client
Once you've added your selected models LiteLLM allows you to make `completion` calls

```python
import litellm
from litellm import completion
litellm.token = "80888ede-4881-4876-ab3f-765d47282e66" # set your token 
messages = [{ "content": "Hello, how are you?" ,"role": "user"}]

# no need to set key, LiteLLM Client reads your set key 
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```


## LiteLLM Dashboard - Debug Logs 
All your `completion()` and `embedding()` call logs are available on `admin.litellm.ai/<your-token>`


### Debug Logs for `completion()` and `embedding()`
<Image img={require('../../img/lite_logs.png')} alt="Dashboard" />

### Viewing Errors on debug logs
<Image img={require('../../img/lite_logs2.png')} alt="Dashboard" />



