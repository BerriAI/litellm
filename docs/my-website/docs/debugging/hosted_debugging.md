import Image from '@theme/IdealImage';
import QueryParamReader from '../../src/components/queryParamReader.js'

# Debug + Deploy LLMs [UI]

LiteLLM offers a UI to:
* 1-Click Deploy LLMs - the client stores your api keys + model configurations
* Debug your Call Logs 

👉 Jump to our sample LiteLLM Dashboard: https://admin.litellm.ai/


<Image img={require('../../img/alt_dashboard.png')} alt="Dashboard" />

## Debug your first logs
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_OpenAI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### 1. Make a normal `completion()` call

```
pip install litellm
```

<QueryParamReader/>

### 2. Check request state
All `completion()` calls print with a link to your session dashboard

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

## Deploy your first LLM

LiteLLM also lets you to add a new model to your project - without touching code **or** using a proxy server. 

### 1. Add new model
On the same debugger dashboard we just made, just go to the 'Add New LLM' Section:
* Select Provider
* Select your LLM 
* Add your LLM Key

<Image img={require('../../img/add_model.png')} alt="Dashboard" />  

This works with any model on - Replicate, Together_ai, Baseten, Anthropic, Cohere, AI21, OpenAI, Azure, VertexAI (Google Palm), OpenRouter

After adding your new LLM, LiteLLM securely stores your API key and model configs. 

[👋 Tell us if you need to self-host **or** integrate with your key manager](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version?month=2023-08)  


### 2. Test new model Using `completion()`
Once you've added your models LiteLLM completion calls will just work for those models + providers.

```python
import litellm
from litellm import completion
litellm.token = "80888ede-4881-4876-ab3f-765d47282e66" # use your token 
messages = [{ "content": "Hello, how are you?" ,"role": "user"}]

# no need to set key, LiteLLM Client reads your set key 
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

###  3. [Bonus] Get available model list

Get a list of all models you've created through the Dashboard with 1 function call 

```python 
import litellm 

litellm.token = "80888ede-4881-4876-ab3f-765d47282e66" # use your token 

litellm.get_model_list()
```
## Persisting your dashboard
If you want to use the same dashboard for your project set
`litellm.token` in code or your .env as `LITELLM_TOKEN`
All generated dashboards come with a token
```python
import litellm
litellm.token = "80888ede-4881-4876-ab3f-765d47282e66"
```


## Additional Information
### LiteLLM Dashboard - Debug Logs 
All your `completion()` and `embedding()` call logs are available on `admin.litellm.ai/<your-token>`


#### Debug Logs for `completion()` and `embedding()`
<Image img={require('../../img/lite_logs.png')} alt="Dashboard" />

#### Viewing Errors on debug logs
<Image img={require('../../img/lite_logs2.png')} alt="Dashboard" />


### Opt-Out of using LiteLLM Client
If you want to opt out of using LiteLLM client you can set
```python
litellm.use_client = True
```






