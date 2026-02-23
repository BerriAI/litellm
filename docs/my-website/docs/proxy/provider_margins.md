# Fee/Price Margin on LLM Costs

Apply percentage-based or fixed-amount margins to specific providers or globally. This is useful for enterprises that need to add operational overhead costs to bill internal consumers.

## When to Use This Feature

If your Generative AI platform involves various operational and architectural overheads, along with infrastructure costs, you may need the capability to apply an additional fee or margin to the total LLM costs. 

**Common use cases:**
- **Internal chargebacks** - Add operational overhead costs when billing internal teams
- **Cost recovery** - Recover infrastructure, support, and platform maintenance costs

## Setup Margins via UI

This walkthrough shows how to add a provider margin and view the cost breakdown in the LiteLLM UI.

### Step 1: Navigate to Settings

From the LiteLLM dashboard, click on **Settings** in the left sidebar.

![Click Settings](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/a9a42382-1c93-4338-8c7e-c0ebc4ee239f/ascreenshot.jpeg?tl_px=0,730&br_px=2064,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=47,292)

### Step 2: Open Cost Tracking

Click on **Cost Tracking** to access the cost configuration options.

![Click Cost Tracking](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/c3ad52c0-1c8d-4be5-bd04-1e37ce186c8e/ascreenshot.jpeg?tl_px=0,730&br_px=2064,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=65,403)

### Step 3: Select Fee/Price Margin

Click on **Fee/Price Margin** - this section allows you to add fees or margins to LLM costs for internal billing and cost recovery.

![Click Fee/Price Margin](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/0810c7bf-e927-4ab6-a55d-37c51d8c17af/ascreenshot.jpeg?tl_px=553,0&br_px=2618,1153&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=551,220)

### Step 4: Add Provider Margin

Click **+ Add Provider Margin** to create a new margin configuration.

![Click Add Provider Margin](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/8762b7d9-74e5-45eb-acc3-be0d9c5b799d/ascreenshot.jpeg?tl_px=553,2&br_px=2618,1155&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=929,277)

### Step 5: Select Provider

Click the search field to select which provider to apply the margin to.

![Click search field](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/7ff01cdc-2749-43f3-a46f-4fd5543446e3/ascreenshot.jpeg?tl_px=507,0&br_px=2572,1153&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,177)

You can select **Global (All Providers)** to apply the margin to all providers, or choose a specific provider like Bedrock, OpenAI, or Anthropic.

![Select Global](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/c9efe187-0995-45ae-9366-290cb20835a2/ascreenshot.jpeg?tl_px=0,0&br_px=2064,1153&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=485,182)

In this example, we'll select **Bedrock** as the provider.

![Select Bedrock](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/ea1524ed-7217-4ee6-9beb-797e3ff08b3a/ascreenshot.jpeg?tl_px=0,0&br_px=2617,1462&force_format=jpeg&q=100&width=1120.0)

### Step 6: Choose Margin Type

Select the margin type. You can choose between **Percentage-based** (e.g., 10% markup) or **Fixed Amount** (e.g., $0.001 per request).

![Click Percentage-based](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/137ffea5-0a5e-445a-809f-a85d20701c87/ascreenshot.jpeg?tl_px=0,0&br_px=2064,1153&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=355,259)

For this example, we'll select **Fixed Amount** to add a flat fee per request.

![Click Fixed Amount](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/56828562-2bae-4f69-b68e-13b1b6a03aa6/ascreenshot.jpeg?tl_px=0,0&br_px=2064,1153&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=493,252)

### Step 7: Enter Margin Value

Enter the margin value. In this example, we're adding a $25 fixed fee per request.

![Enter margin value](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/80018d4b-0205-43a3-a534-9a0e39ddf139/ascreenshot.jpeg?tl_px=0,0&br_px=2618,1462&force_format=jpeg&q=100&width=1120.0)

### Step 8: Save the Margin

Click **Add Provider Margin** to save your configuration.

![Click Add Provider Margin](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/84a5bcb8-f475-4aef-83ec-f0b3b620613f/ascreenshot.jpeg?tl_px=553,206&br_px=2618,1359&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=636,276)

### Step 9: Test the Margin in Playground

Navigate to **Playground** to test your margin configuration by making a request.

![Click Playground](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/cda7293a-2439-4301-bc44-211e6d6833a6/ascreenshot.jpeg?tl_px=0,0&br_px=2064,1153&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=37,106)

Select a model and send a test message.

![Send test message](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/48c3e28e-a01a-483c-838d-2d1643f44be7/ascreenshot.jpeg?tl_px=0,0&br_px=2617,1462&force_format=jpeg&q=100&width=1120.0)

Enter your prompt in the message field and submit.

![Enter prompt](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/88963dbe-6bad-4aac-8bd3-7f4eac0dd995/ascreenshot.jpeg?tl_px=243,730&br_px=2308,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,451)

You'll receive a response from the model.

![View response](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/1d69ef9c-cc22-40ad-8f10-f14a359d2fb6/ascreenshot.jpeg?tl_px=553,17&br_px=2618,1170&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=549,276)

### Step 10: View Cost Breakdown in Logs

Navigate to **Logs** to view the detailed cost breakdown for your request.

![Click Logs](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/5cf6dd8b-0783-41ee-b23a-32f3424c2092/ascreenshot.jpeg?tl_px=0,99&br_px=2064,1252&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=32,276)

Click on the expand icon to view the request details.

![Click expand icon](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/3ae2900f-1515-4bb9-a4aa-328b43f13b61/ascreenshot.jpeg?tl_px=0,12&br_px=2064,1165&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=187,277)

### Step 11: View Cost Breakdown Details

Click on **Cost Breakdown** to see how the total cost was calculated, including the margin.

![Click Cost Breakdown](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/8bce9050-58ca-4860-9e18-1b704e086cf4/ascreenshot.jpeg?tl_px=392,575&br_px=2457,1728&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,276)

The cost breakdown shows the margin amount that was added. In this example, you can see the **+$25.00** margin clearly displayed.

![View margin amount](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/c4a65d38-a47a-4634-baf2-608447a7d711/ascreenshot.jpeg?tl_px=0,730&br_px=2064,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=388,282)

The total cost reflects the base LLM cost plus the margin, giving you full transparency into your cost structure.

![View total cost](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-25/3b13550d-5255-4818-b3ee-3d4391991c13/ascreenshot.jpeg?tl_px=0,730&br_px=2064,1884&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=384,323)

## Setup Margins via Config

You can also configure margins directly in your `config.yaml` file.

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
