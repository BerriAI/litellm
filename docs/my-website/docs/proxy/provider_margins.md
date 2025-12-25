# Provider Margins

Apply percentage-based or fixed-amount margins to specific providers or globally. This is useful for enterprises that need to add operational overhead costs to bill internal consumers.

## Usage with LiteLLM Proxy Server

**Step 1: Add margin config to config.yaml**

```yaml
# Apply margins to providers
cost_margin_config:
  global: 0.05            # 5% global margin on all providers
  openai: 0.10            # 10% margin for OpenAI (overrides global)
  anthropic:
    fixed_amount: 0.001   # $0.001 fixed fee per request
```

**Step 2: Start proxy**

```bash
litellm /path/to/config.yaml
```

The margin will be automatically applied to all cost calculations for the configured providers.

## How Margins Work

- Margins are applied **after** discounts (if configured)
- Margins are calculated independently from discounts
- You can use:
  - **Percentage-based**: `{"openai": 0.10}` = 10% margin
  - **Fixed amount**: `{"openai": {"fixed_amount": 0.001}}` = $0.001 per request
  - **Global**: `{"global": 0.05}` = 5% margin on all providers (unless provider-specific margin exists)
- Provider-specific margins override global margins
- Margin information is tracked in cost breakdown logs
- Margin information is returned in response headers:
  - `x-litellm-response-cost-margin-amount` - Total margin added in USD
  - `x-litellm-response-cost-margin-percent` - Margin percentage applied

## Margin Calculation Examples

**Example 1: Percentage-only margin**
```yaml
cost_margin_config:
  openai: 0.10  # 10% margin
```
If base cost is $1.00, final cost = $1.00 x 1.10 = $1.10

**Example 2: Fixed amount only**
```yaml
cost_margin_config:
  anthropic:
    fixed_amount: 0.001  # $0.001 per request
```
If base cost is $1.00, final cost = $1.00 + $0.001 = $1.001

**Example 3: Global margin with provider override**
```yaml
cost_margin_config:
  global: 0.05   # 5% global margin
  openai: 0.10   # 10% margin for OpenAI (overrides global)
```
- OpenAI requests: 10% margin applied
- All other providers: 5% margin applied

## Margins with Discounts

Margins and discounts are calculated independently:

1. Base cost is calculated
2. Discount is applied (if configured)
3. Margin is applied to the discounted cost

**Example:**
```yaml
cost_discount_config:
  openai: 0.05  # 5% discount
cost_margin_config:
  openai: 0.10  # 10% margin
```

If base cost is $1.00:
- After discount: $1.00 x 0.95 = $0.95
- After margin: $0.95 x 1.10 = $1.045

## Supported Providers

You can apply margins to all LiteLLM supported providers, or use `global` to apply to all providers. Common examples:

- `global` - Applies to all providers (unless provider-specific margin exists)
- `openai` - OpenAI
- `anthropic` - Anthropic
- `vertex_ai` - Google Vertex AI
- `gemini` - Google Gemini
- `azure` - Azure OpenAI
- `bedrock` - AWS Bedrock

See the full list of providers in the [LlmProviders](https://github.com/BerriAI/litellm/blob/main/litellm/types/utils.py) enum.

