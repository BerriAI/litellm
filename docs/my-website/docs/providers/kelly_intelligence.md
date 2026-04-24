import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Kelly Intelligence
https://api.thedailylesson.com

Kelly Intelligence is an OpenAI-compatible AI tutor API powered by Claude with automatic
vocabulary RAG over 162,000 words across 47 languages. Built by
[Lesson of the Day, PBC](https://lotdpbc.com), a public benefit corporation.

| Property | Details |
|---------|--------|
| Description | OpenAI-compatible AI tutor with vocabulary RAG (162K words, 47 languages) |
| Provider Route | `openai/` (uses LiteLLM's OpenAI-compatible client) |
| Provider Doc | [Kelly Intelligence ↗](https://api.thedailylesson.com) |
| API Endpoint | `https://api.thedailylesson.com/v1` |
| Free Tier | 500 calls/month, no credit card |

## API Key

Get a free API key at https://api.thedailylesson.com (500 calls/month, no credit card).

```python
# env variable
os.environ['KELLY_API_KEY']
```

## Sample Usage

```python
from litellm import completion
import os

os.environ['KELLY_API_KEY'] = ""
response = completion(
    model="openai/kelly-haiku",
    api_base="https://api.thedailylesson.com/v1",
    api_key=os.environ['KELLY_API_KEY'],
    messages=[{"role": "user", "content": "What does ephemeral mean?"}],
)
print(response)
```

## Sample Usage - Streaming

```python
from litellm import completion
import os

os.environ['KELLY_API_KEY'] = ""
response = completion(
    model="openai/kelly-sonnet",
    api_base="https://api.thedailylesson.com/v1",
    api_key=os.environ['KELLY_API_KEY'],
    messages=[{"role": "user", "content": "Teach me about serendipity"}],
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Usage with LiteLLM Proxy Server

1. Setup config.yaml

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
  - model_name: kelly-opus
    litellm_params:
      model: openai/kelly-opus
      api_base: https://api.thedailylesson.com/v1
      api_key: os.environ/KELLY_API_KEY
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer anything" \
  -d '{
    "model": "kelly-haiku",
    "messages": [{"role": "user", "content": "What does ubiquitous mean?"}]
  }'
```

## Supported Models

| Model Name     | Engine            | Best For                  | Min Tier |
|----------------|-------------------|---------------------------|----------|
| kelly-haiku    | Claude Haiku 4.5  | Fast tutoring, quizzes    | Free     |
| kelly-sonnet   | Claude Sonnet 4.6 | Deep explanations         | Developer |
| kelly-opus     | Claude Opus 4.6   | Curriculum design         | Pro      |
| claude-haiku   | Claude Haiku 4.5  | Raw Claude + vocab RAG    | Free     |
| claude-sonnet  | Claude Sonnet 4.6 | Raw Claude + vocab RAG    | Developer |
| claude-opus    | Claude Opus 4.6   | Raw Claude + vocab RAG    | Pro      |

`kelly-*` models include the Kelly tutor persona with a 5-phase Socratic teaching method.
`claude-*` models are raw Claude with vocabulary RAG only.

## What's Different From Raw Claude?

- **162K words** auto-injected as RAG context (definitions, etymology, IPA)
- **601K translations** across 47 languages
- **AI tutor persona** (Kelly) — warm, Socratic, 5-phase teaching method
- **OpenAI-compatible** format — works with any OpenAI client
- **Free tier** — 500 calls/month, no credit card

## API Reference

- OpenAPI 3.1 spec: https://api.thedailylesson.com/openapi.json
- AI plugin manifest: https://api.thedailylesson.com/.well-known/ai-plugin.json
- Docs: https://api.thedailylesson.com
