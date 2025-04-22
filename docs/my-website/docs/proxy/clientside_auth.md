# Clientside LLM Credentials 


### Pass User LLM API Keys, Fallbacks
Allow your end-users to pass their model list, api base, OpenAI API key (any LiteLLM supported provider) to make requests 

**Note** This is not related to [virtual keys](./virtual_keys.md). This is for when you want to pass in your users actual LLM API keys. 

:::info

**You can pass a litellm.RouterConfig as `user_config`, See all supported params here https://github.com/BerriAI/litellm/blob/main/litellm/types/router.py **

:::

<Tabs>

<TabItem value="openai-py" label="OpenAI Python">

#### Step 1: Define user model list & config
```python
import os

user_config = {
    'model_list': [
        {
            'model_name': 'user-azure-instance',
            'litellm_params': {
                'model': 'azure/chatgpt-v-2',
                'api_key': os.getenv('AZURE_API_KEY'),
                'api_version': os.getenv('AZURE_API_VERSION'),
                'api_base': os.getenv('AZURE_API_BASE'),
                'timeout': 10,
            },
            'tpm': 240000,
            'rpm': 1800,
        },
        {
            'model_name': 'user-openai-instance',
            'litellm_params': {
                'model': 'gpt-3.5-turbo',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'timeout': 10,
            },
            'tpm': 240000,
            'rpm': 1800,
        },
    ],
    'num_retries': 2,
    'allowed_fails': 3,
    'fallbacks': [
        {
            'user-azure-instance': ['user-openai-instance']
        }
    ]
}


```

#### Step 2: Send user_config in `extra_body`
```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# send request to `user-azure-instance`
response = client.chat.completions.create(model="user-azure-instance", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
], 
    extra_body={
      "user_config": user_config
    }
) # 👈 User config

print(response)
```

</TabItem>

<TabItem value="openai-js" label="OpenAI JS">

#### Step 1: Define user model list & config
```javascript
const os = require('os');

const userConfig = {
    model_list: [
        {
            model_name: 'user-azure-instance',
            litellm_params: {
                model: 'azure/chatgpt-v-2',
                api_key: process.env.AZURE_API_KEY,
                api_version: process.env.AZURE_API_VERSION,
                api_base: process.env.AZURE_API_BASE,
                timeout: 10,
            },
            tpm: 240000,
            rpm: 1800,
        },
        {
            model_name: 'user-openai-instance',
            litellm_params: {
                model: 'gpt-3.5-turbo',
                api_key: process.env.OPENAI_API_KEY,
                timeout: 10,
            },
            tpm: 240000,
            rpm: 1800,
        },
    ],
    num_retries: 2,
    allowed_fails: 3,
    fallbacks: [
        {
            'user-azure-instance': ['user-openai-instance']
        }
    ]
};
```

#### Step 2: Send `user_config` as a param to `openai.chat.completions.create`

```javascript
const { OpenAI } = require('openai');

const openai = new OpenAI({
  apiKey: "sk-1234",
  baseURL: "http://0.0.0.0:4000"
});

async function main() {
  const chatCompletion = await openai.chat.completions.create({
    messages: [{ role: 'user', content: 'Say this is a test' }],
    model: 'gpt-3.5-turbo',
    user_config: userConfig // # 👈 User config
  });
}

main();
```

</TabItem>

</Tabs>

### Pass User LLM API Keys / API Base
Allows your users to pass in their OpenAI API key/API base (any LiteLLM supported provider) to make requests 

Here's how to do it: 

#### 1. Enable configurable clientside auth credentials for a provider

```yaml
model_list:
  - model_name: "fireworks_ai/*"
    litellm_params:
      model: "fireworks_ai/*"
      configurable_clientside_auth_params: ["api_base"]
      # OR 
      configurable_clientside_auth_params: [{"api_base": "^https://litellm.*direct\.fireworks\.ai/v1$"}] # 👈 regex
```

Specify any/all auth params you want the user to be able to configure:

- api_base (✅ regex supported)
- api_key
- base_url 

(check [provider docs](../providers/) for provider-specific auth params - e.g. `vertex_project`)


#### 2. Test it!

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
], 
    extra_body={"api_key": "my-bad-key", "api_base": "https://litellm-dev.direct.fireworks.ai/v1"}) # 👈 clientside credentials

print(response)
```

More examples: 
<Tabs>
<TabItem value="openai-py" label="Azure Credentials">

Pass in the litellm_params (E.g. api_key, api_base, etc.) via the `extra_body` parameter in the OpenAI client. 

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
], 
    extra_body={
      "api_key": "my-azure-key",
      "api_base": "my-azure-base",
      "api_version": "my-azure-version"
    }) # 👈 User Key

print(response)
```


</TabItem>
<TabItem value="openai-js" label="OpenAI JS">

For JS, the OpenAI client accepts passing params in the `create(..)` body as normal.

```javascript
const { OpenAI } = require('openai');

const openai = new OpenAI({
  apiKey: "sk-1234",
  baseURL: "http://0.0.0.0:4000"
});

async function main() {
  const chatCompletion = await openai.chat.completions.create({
    messages: [{ role: 'user', content: 'Say this is a test' }],
    model: 'gpt-3.5-turbo',
    api_key: "my-bad-key" // 👈 User Key
  });
}

main();
```
</TabItem>
</Tabs>

### Pass provider-specific params (e.g. Region, Project ID, etc.)

Specify the region, project id, etc. to use for making requests to Vertex AI on the clientside.

Any value passed in the Proxy's request body, will be checked by LiteLLM against the mapped openai / litellm auth params. 

Unmapped params, will be assumed to be provider-specific params, and will be passed through to the provider in the LLM API's request body.

```bash
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={ # pass any additional litellm_params here
        vertex_ai_location: "us-east1" 
    }
)

print(response)
```