import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# NoPII

## Overview

| Property | Details |
|-------|-------|
| Description | NoPII is a hosted privacy proxy for LLM APIs: personal data (names, SSNs, emails, phone numbers, credit cards, etc.) in outbound requests is substituted with deterministic vault tokens, and restored in the response. The LLM provider only ever sees tokenized placeholders, never raw PII. This shrinks exposure from provider-side breaches, logging mishaps, and subpoenas, and narrows the scope of third-party data-processing obligations. |
| Integration Pattern | Drop-in proxy via `api_base` override. Use your existing `openai/` or `anthropic/` model strings; just redirect the base URL. No NoPII-specific model prefix. |
| Link to Provider Doc | [NoPII Documentation ↗](https://docs.nopii.co/quickstart) |
| Base URL | `https://api.nopii.co` |
| Supported Operations | `/v1/chat/completions` (OpenAI-compatible), `/v1/messages` (Anthropic-compatible), streaming on both |

<br />

**NoPII works with any model available through OpenAI's chat-completions API or Anthropic's messages API. Point LiteLLM at `https://api.nopii.co` via the [`api_base`](../completion/input) parameter and continue using the `openai/` or `anthropic/` routes you already use.**

## What is NoPII?

NoPII sits between your application and the LLM provider. The proxy detects entities in your request, vaults the plaintext, and forwards only opaque tokens (e.g. `[NAME: VAULT_72GleckHu]`) to the model. When the model responds, NoPII restores the original values before returning the response to you. The label is a neutralized category (`NAME`, `EMAIL`, `IDENTIFIER`, `PHONE`, `ADDRESS`, `LOCATION`), not the raw entity type, which prevents the model from refusing to process content tagged with sensitive types like SSN or credit card.

Tokens are deterministic, so multi-turn conversations stay coherent automatically: the model sees the same token for the same entity in every turn it appears.

NoPII is a hosted service; no self-hosting or extra SDK is required.

## Required Variables

NoPII identifies your tenant from your LLM provider key (via a one-way hash), so no separate NoPII credential is required.

```python showLineNumbers title="Environment Variables"
import os

os.environ["OPENAI_API_KEY"] = ""     # your OpenAI key (for openai/* models)
os.environ["ANTHROPIC_API_KEY"] = ""  # your Anthropic key (for anthropic/* models)
```

Sign up and configure your detection settings at [app.nopii.co](https://app.nopii.co).

## Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="openai" label="OpenAI-compatible">

```python showLineNumbers title="OpenAI via NoPII"
import os
import litellm

response = litellm.completion(
    model="openai/gpt-5",
    api_key=os.environ["OPENAI_API_KEY"],
    api_base="https://api.nopii.co",
    messages=[
        {
            "role": "user",
            "content": (
                "Summarize the customer record for Sarah Chen. "
                "Her SSN is 456-78-9012 and her email is sarah.chen@example.com."
            ),
        }
    ],
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic-compatible">

```python showLineNumbers title="Anthropic via NoPII"
import os
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    api_base="https://api.nopii.co",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": (
                "Summarize the customer record for Sarah Chen. "
                "Her SSN is 456-78-9012 and her email is sarah.chen@example.com."
            ),
        }
    ],
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

### Streaming

<Tabs>
<TabItem value="openai-stream" label="OpenAI-compatible">

```python showLineNumbers title="OpenAI streaming via NoPII"
import os
import litellm

response = litellm.completion(
    model="openai/gpt-5",
    api_key=os.environ["OPENAI_API_KEY"],
    api_base="https://api.nopii.co",
    messages=[{"role": "user", "content": "Write a short poem about privacy."}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

</TabItem>
<TabItem value="anthropic-stream" label="Anthropic-compatible">

```python showLineNumbers title="Anthropic streaming via NoPII"
import os
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    api_base="https://api.nopii.co",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a short poem about privacy."}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

</TabItem>
</Tabs>

## Usage - LiteLLM Proxy (config.yaml)

Route chat-completions (OpenAI) or messages (Anthropic) models through NoPII by setting `api_base` on the model entry. The rest of your proxy config is unchanged.

```yaml title="config.yaml"
model_list:
  - model_name: gpt-5-nopii
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.nopii.co

  - model_name: claude-sonnet-4-6-nopii
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
      api_base: https://api.nopii.co
```

Call the proxy as you would any other model:

```bash title="Proxy request"
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "gpt-5-nopii",
    "messages": [{"role": "user", "content": "My email is alex@example.com"}]
  }'
```

## Sessions (optional)

Pass an `X-NoPII-Session-Id` header to group multi-turn conversations in your audit log and skip redundant vault calls on repeated PII. Tokenization is deterministic regardless, so multi-turn conversations stay coherent without it.

```python showLineNumbers title="Session header via LiteLLM"
import os
import litellm

response = litellm.completion(
    model="openai/gpt-5",
    api_key=os.environ["OPENAI_API_KEY"],
    api_base="https://api.nopii.co",
    extra_headers={"X-NoPII-Session-Id": "my-session-123"},
    messages=[{"role": "user", "content": "..."}],
)
```

See [Sessions](https://docs.nopii.co/sessions) for details.

## Supported Entity Types

NoPII detects 35+ entity types covering personal information (names, emails, phones), government IDs (SSN, ITIN, passport, driver's license, medical license, and more), financial data (credit cards, bank accounts, IBAN, crypto wallets), network identifiers (IP, MAC, URL), locations, dates, and credentials. The enabled set and detection confidence threshold (default `0.4`) are configurable per tenant in the [NoPII admin console](https://app.nopii.co). See the [NoPII docs](https://docs.nopii.co) for the full list.

## Additional Resources

- [NoPII website](https://nopii.co)
- [NoPII docs](https://docs.nopii.co)
- [LiteLLM examples](https://github.com/Enigma-Vault/NoPII-Examples/tree/main/examples/litellm)
