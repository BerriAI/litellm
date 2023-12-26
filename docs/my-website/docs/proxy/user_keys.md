import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Pass in User Keys

Send user keys to the proxy


Here's how to do it: 

<Tabs>
<TabItem value="openai-py" label="OpenAI Python">

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
    extra_body={"api_key": "my-bad-key"}) # 👈 User Key

print(response)
```
</TabItem>
<TabItem value="openai-js" label="OpenAI JS">

```javascript
const { OpenAI } = require('openai');

const openai = new OpenAI({
  apiKey: "sk-1234", // This is the default and can be omitted
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