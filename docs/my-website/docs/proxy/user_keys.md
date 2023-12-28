import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Pass User LLM API Keys

Allows your users to pass in their OpenAI API key (any LiteLLM supported provider) to make requests 


Here's how to do it: 

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:8000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
], 
    extra_body={"api_key": "my-bad-key"}) # 👈 User Key

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
    base_url="http://0.0.0.0:8000"
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
  baseURL: "http://0.0.0.0:8000"
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