# Provider Discounts

Apply percentage-based discounts to specific providers. This is useful for negotiated enterprise pricing with providers.

## Usage with LiteLLM Proxy Server

**Step 1: Add discount config to config.yaml**

```yaml
# Apply 5% discount to all Vertex AI and Gemini costs
cost_discount_config:
  vertex_ai: 0.05  # 5% discount
  gemini: 0.05     # 5% discount
  openrouter: 0.05 # 5% discount
  # openai: 0.10   # 10% discount (example)
```

**Step 2: Start proxy**

```bash
litellm /path/to/config.yaml
```

The discount will be automatically applied to all cost calculations for the configured providers.


## How Discounts Work

- Discounts are applied **after** all other cost calculations (tokens, caching, tools, etc.)
- The discount is a percentage (0.05 = 5%, 0.10 = 10%, etc.)
- Discounts only apply to the configured providers
- Original cost, discount amount, and final cost are tracked in cost breakdown logs
- Discount information is returned in response headers:
  - `x-litellm-response-cost` - Final cost after discount
  - `x-litellm-response-cost-original` - Cost before discount
  - `x-litellm-response-cost-discount-amount` - Discount amount in USD

## Supported Providers

You can apply discounts to all LiteLLM supported providers. Common examples:

- `vertex_ai` - Google Vertex AI
- `gemini` - Google Gemini
- `openai` - OpenAI
- `anthropic` - Anthropic
- `azure` - Azure OpenAI
- `bedrock` - AWS Bedrock
- `cohere` - Cohere
- `openrouter` - OpenRouter

See the full list of providers in the [LlmProviders](https://github.com/BerriAI/litellm/blob/main/litellm/types/utils.py) enum.

