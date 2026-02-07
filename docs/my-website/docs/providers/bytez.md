import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bytez

LiteLLM supports all chat models on [Bytez](https://www.bytez.com)!

That also means multi-modal models are supported 🔥

Tasks supported: `chat`, `image-text-to-text`, `audio-text-to-text`, `video-text-to-text`

## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

### API KEYS

```py
import os
os.environ["BYTEZ_API_KEY"] = "YOUR_BYTEZ_KEY_GOES_HERE"
```

### Example Call

```py
from litellm import completion
import os
## set ENV variables
os.environ["BYTEZ_API_KEY"] = "YOUR_BYTEZ_KEY_GOES_HERE"

response = completion(
    model="bytez/google/gemma-3-4b-it",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add models to your config.yaml

```yaml
model_list:
  - model_name: gemma-3
    litellm_params:
      model: bytez/google/gemma-3-4b-it
      api_key: os.environ/BYTEZ_API_KEY
```

2. Start the proxy

```bash
$ BYTEZ_API_KEY=YOUR_BYTEZ_API_KEY_HERE litellm --config /path/to/config.yaml --debug
```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

```py
import openai
client = openai.OpenAI(
    api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:4000" # litellm-proxy-base url
)

response = client.chat.completions.create(
    model="gemma-3",
    messages = [
      {
          "role": "system",
          "content": "Be a good human!"
      },
      {
          "role": "user",
          "content": "What do you know about earth?"
      }
  ]
)

print(response)
```

  </TabItem>

  <TabItem value="curl" label="curl">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gemma-3",
    "messages": [
      {
          "role": "system",
          "content": "Be a good human!"
      },
      {
          "role": "user",
          "content": "What do you know about earth?"
      }
      ],
}'
```

  </TabItem>

  </Tabs>

</TabItem>

</Tabs>

## Automatic Prompt Template Handling

All prompt formatting is handled automatically by our API when you send a messages list to it!

If you wish to use custom formatting, please let us know via either [help@bytez.com](mailto:help@bytez.com) or on our [Discord](https://discord.com/invite/Z723PfCFWf) and we will work to provide it!

## Passing additional params - max_tokens, temperature

See all litellm.completion supported params [here](https://docs.litellm.ai/docs/completion/input)

```py
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["BYTEZ_API_KEY"] = "YOUR_BYTEZ_KEY_HERE"

# bytez gemma-3 call
response = completion(
    model="bytez/google/gemma-3-4b-it",
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    max_tokens=20,
    temperature=0.5
)
```

**proxy**

```yaml
model_list:
  - model_name: gemma-3
    litellm_params:
      model: bytez/google/gemma-3-4b-it
      api_key: os.environ/BYTEZ_API_KEY
      max_tokens: 20
      temperature: 0.5
```

## Passing Bytez-specific params

Any kwarg supported by huggingface we also support! (Provided the model supports it.)

Example `repetition_penalty`

```py
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["BYTEZ_API_KEY"] = "YOUR_BYTEZ_KEY_HERE"

# bytez llama3 call with additional params
response = completion(
    model="bytez/google/gemma-3-4b-it",
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    repetition_penalty=1.2,
)
```

**proxy**

```yaml
model_list:
  - model_name: gemma-3
    litellm_params:
      model: bytez/google/gemma-3-4b-it
      api_key: os.environ/BYTEZ_API_KEY
      repetition_penalty: 1.2
```
