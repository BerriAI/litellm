import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LiteLLM Native PII Guardrail

LiteLLM provides built-in PII (Personally Identifiable Information) detection and masking capabilities powered by [Presidio](https://github.com/microsoft/presidio).

## Quick Start
### 1. Define Guardrails in your config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "pii-guard"
    litellm_params:
      guardrail: litellm_pii
      mode: "pre_call"
      default_on: true
      entities_config:
        PHONE_NUMBER: "MASK"
        EMAIL_ADDRESS: "MASK"
        PERSON: "MASK"
        LOCATION: "BLOCK"
        CREDIT_CARD: "BLOCK"
```

## Supported PII Entity Types

The `entities_config` parameter defines which PII entities to detect and how to handle them:

```yaml
entities_config:
  ENTITY_TYPE: "ACTION"
```

### Available Entity Types
- `PHONE_NUMBER`: Phone numbers (e.g., 212-555-5555)
- `EMAIL_ADDRESS`: Email addresses (e.g., user@example.com)
- `PERSON`: Person names (e.g., John Doe)
- `LOCATION`: Geographical locations (e.g., New York)
- `URL`: Web URLs (e.g., https://example.com)
- `CREDIT_CARD`: Credit card numbers (e.g., 1234-5678-9012-3456)

### Available Actions
- `MASK`: Replace detected PII with a placeholder tag (e.g., `<EMAIL_ADDRESS>`)
- `BLOCK`: Block the request entirely if any specified PII is detected

## Example Usage

<Tabs>
<TabItem label="Masking PII" value="masking">

```yaml
guardrails:
  - guardrail_name: "mask-pii"
    litellm_params:
      guardrail: litellm_pii
      mode: "pre_call"
      entities_config:
        PHONE_NUMBER: "MASK"
        EMAIL_ADDRESS: "MASK"
```

With this configuration, a user message like:
```
My email is john@example.com and my phone is 555-123-4567
```

Will be transformed to:
```
My email is <EMAIL_ADDRESS> and my phone is <PHONE_NUMBER>
```

</TabItem>

<TabItem label="Blocking PII" value="blocking">

```yaml
guardrails:
  - guardrail_name: "block-pii"
    litellm_params:
      guardrail: litellm_pii
      mode: "pre_call"
      entities_config:
        CREDIT_CARD: "BLOCK"
        LOCATION: "BLOCK"
```

If a user sends a message containing a credit card number or location, the request will be blocked entirely, and an error response will be returned.

</TabItem>
</Tabs>

## Modes

- `pre_call`: Run on user input before sending to the LLM
- `post_call`: Run on LLM output before returning to the user
- `during_call`: Run in parallel with the LLM call 