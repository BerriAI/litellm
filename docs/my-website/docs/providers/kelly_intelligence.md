import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Kelly Intelligence
https://api.thedailylesson.com

[Kelly Intelligence](https://api.thedailylesson.com) is an OpenAI-compatible API
with a built-in 162,000-word vocabulary RAG layer and an opinionated AI tutor
persona ("Kelly"), built on top of Claude. It's operated by
[Lesson of the Day, PBC](https://lotdpbc.com), a public benefit corporation
building education infrastructure. The free tier includes 500 calls per month
with no credit card required.

:::tip

Kelly Intelligence is OpenAI-compatible, so you can call it via LiteLLM's
`openai/` provider with a custom `api_base`. No core LiteLLM changes required.

:::

## API Key

Get a free API key (no credit card) at [api.thedailylesson.com](https://api.thedailylesson.com).

```python
import os

os.environ["KELLY_API_KEY"] = "your-kelly-api-key"
```

## Sample Usage

```python
from litellm import completion
import os

os.environ["KELLY_API_KEY"] = ""

response = completion(
    model="openai/kelly-haiku",
    api_key=os.environ["KELLY_API_KEY"],
    api_base="https://api.thedailylesson.com/v1",
    messages=[
        {"role": "user", "content": "Teach me the word 'serendipity' in one paragraph."}
    ],
)

print(response)
```

## Sample Usage - Streaming

```python
from litellm import completion
import os

os.environ["KELLY_API_KEY"] = ""

response = completion(
    model="openai/kelly-haiku",
    api_key=os.environ["KELLY_API_KEY"],
    api_base="https://api.thedailylesson.com/v1",
    messages=[
        {"role": "user", "content": "Teach me the word 'serendipity' in one paragraph."}
    ],
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Try without an API key

Kelly Intelligence has a public `/v1/demo` endpoint for trying the API without
any signup, IP-rate-limited at 5 requests per hour:

```bash
curl -X POST https://api.thedailylesson.com/v1/demo \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What does ephemeral mean?"}]}'
```

This is the same wire format as `/v1/chat/completions` — useful for evaluating
the integration before adding it to your config.

## Supported Models

| Model Name     | Tier      | Function Call                                              |
|----------------|-----------|------------------------------------------------------------|
| kelly-haiku    | Free      | `completion(model="openai/kelly-haiku", ...)`              |
| kelly-sonnet   | Developer | `completion(model="openai/kelly-sonnet", ...)`             |
| kelly-opus     | Pro       | `completion(model="openai/kelly-opus", ...)`               |
| claude-haiku   | Free      | `completion(model="openai/claude-haiku", ...)`             |
| claude-sonnet  | Developer | `completion(model="openai/claude-sonnet", ...)`            |
| claude-opus    | Pro       | `completion(model="openai/claude-opus", ...)`              |

The `kelly-*` models include the AI tutor persona; the `claude-*` models route
to Claude with vocabulary RAG only and no persona overlay.

## Usage with LiteLLM Proxy

### 1. Add Kelly Intelligence to `config.yaml`

```yaml
model_list:
  - model_name: kelly-haiku
    litellm_params:
      model: openai/kelly-haiku
      api_base: https://api.thedailylesson.com/v1
      api_key: os.environ/KELLY_API_KEY

  - model_name: kelly-sonnet
    litellm_params:
      model: openai/kelly-sonnet
      api_base: https://api.thedailylesson.com/v1
      api_key: os.environ/KELLY_API_KEY
```

### 2. Start the proxy

```bash
litellm --config config.yaml
```

### 3. Test it

<Tabs>
<TabItem value="curl" label="Curl">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "kelly-haiku",
    "messages": [
      {"role": "user", "content": "Teach me the word ephemeral."}
    ]
  }'
```

</TabItem>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000",
)

response = client.chat.completions.create(
    model="kelly-haiku",
    messages=[
        {"role": "user", "content": "Teach me the word ephemeral."}
    ],
)
print(response)
```

</TabItem>
</Tabs>

## Bonus: public word lookup endpoint

Kelly Intelligence also exposes a public, no-auth vocabulary lookup endpoint
that returns IPA, definition, etymology, mnemonic, and translations into 47
languages. Useful as a structured RAG source alongside any LiteLLM call:

```bash
curl "https://api.thedailylesson.com/v1/word/serendipity?translations=ES,FR,DE,JA"
```

IP-rate-limited at 60 requests per hour. No API key required.

## Positioning

Kelly Intelligence is **complementary** to direct provider access — not a
replacement. For raw Claude access, the
[Anthropic provider](./anthropic) in LiteLLM is the right tool. Kelly fits when
an application needs the OpenAI wire format **and** vocabulary / language-learning
features without building that data layer in-house.
