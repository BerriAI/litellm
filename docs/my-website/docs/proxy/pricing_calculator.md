# Pricing Calculator (Cost Estimation)

Estimate LLM costs based on expected token usage and request volume. This tool helps developers and platform teams forecast spending before deploying models to production.

## When to Use This Feature

Use the Pricing Calculator to:
- **Budget planning** - Estimate monthly costs before committing to a model
- **Model comparison** - Compare costs across different models for your use case
- **Capacity planning** - Understand cost implications of scaling request volume
- **Cost optimization** - Identify the most cost-effective model for your token requirements

## Using the Pricing Calculator

This walkthrough shows how to estimate LLM costs using the Pricing Calculator in the LiteLLM UI.

### Step 1: Navigate to Settings

From the LiteLLM dashboard, click on **Settings** in the left sidebar.

![Click Settings](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/183c437e-bda9-48b4-ab8f-95f023ba1146/ascreenshot_a1013487f545484194a9a4929eef4c49_text_export.jpeg)

### Step 2: Open Cost Tracking

Click on **Cost Tracking** to access the cost configuration options.

![Click Cost Tracking](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/05c92350-cbae-42ed-935b-e96a26003de8/ascreenshot_cc85f175a6664fc5be8dfdcc1759b442_text_export.jpeg)

### Step 3: Open Pricing Calculator

Click on **Pricing Calculator** to expand the calculator panel. This section allows you to estimate LLM costs based on expected token usage and request volume.

![Click Pricing Calculator](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/31ab5547-fa7d-4abd-b41a-7b4bbc0401f7/ascreenshot_f7f8b098ceba4b5199e5cbc60dddfd0a_text_export.jpeg)

### Step 4: Select a Model

Click the **Model** dropdown to select the model you want to estimate costs for.

![Click Model field](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/a6c236ce-3154-42a8-9701-120e3f7a017b/ascreenshot_635c61b832594e809f8ab79b5b3f32e1_text_export.jpeg)

Choose a model from the list. The models shown are the ones configured on your LiteLLM proxy.

![Select model](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/96c4ebc4-1b88-4dea-b3b2-ea32fde36d9e/ascreenshot_7c2920f05a984ebbb530a8a85e669537_text_export.jpeg)

### Step 5: Configure Token Counts

Enter the expected **Input Tokens (per request)** - this is the average number of tokens in your prompts.

![Click Input Tokens field](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/d0b5ad8a-56e4-4f73-ac66-e1d728c81dc5/ascreenshot_42502082d6204a3891e0a2c3e89a1e38_text_export.jpeg)

Enter the expected **Output Tokens (per request)** - this is the average number of tokens in model responses.

![Click Output Tokens field](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/d7481177-c63c-47f5-9316-1e87695f67f9/ascreenshot_8718cac4c0d14a82ab9f2b71795250c2_text_export.jpeg)

### Step 6: Set Request Volume

Enter your expected request volume. You can specify **Requests per Day** and/or **Requests per Month**.

![Click Requests per Month field](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/42270e11-93f1-41dc-b9c7-3bb6971ced31/ascreenshot_79f2ea9937b34e48ab1ff832ce7f7cb7_text_export.jpeg)

For example, enter `10000000` for 10 million requests per month.

![Enter request volume](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/5e6c4338-ff87-44dd-9059-7577217fa3c8/ascreenshot_15c36610dc914536ac9446470eb39f05_text_export.jpeg)

### Step 7: View Cost Estimates

The calculator automatically updates as you change values. View the cost breakdown including:

- **Per-Request Cost** - Total cost, input cost, output cost, and margin/fee per request
- **Daily Costs** - Aggregated costs if you specified requests per day
- **Monthly Costs** - Aggregated costs if you specified requests per month

![View cost estimates](https://colony-recorder.s3.amazonaws.com/files/2026-01-05/4436cd11-df58-47cb-9742-c0d08865a61c/ascreenshot_f961298a4231464ea841bc4d184f731e_text_export.jpeg)

### Step 8: Export the Report

Click the **Export** button to download your cost estimate. You can export as:

- **PDF** - Opens a print dialog to save as PDF (great for sharing with stakeholders)
- **CSV** - Downloads a spreadsheet-compatible file for further analysis

## Cost Breakdown Details

The Pricing Calculator shows:

| Field | Description |
|-------|-------------|
| **Total Cost** | Complete cost including any configured margins |
| **Input Cost** | Cost for input/prompt tokens |
| **Output Cost** | Cost for output/completion tokens |
| **Margin/Fee** | Any configured [provider margins](/docs/proxy/provider_margins) |
| **Token Pricing** | Per-token rates (shown as $/1M tokens) |

## API Endpoint

You can also estimate costs programmatically using the `/cost/estimate` endpoint:

```bash
curl -X POST "http://localhost:4000/cost/estimate" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "input_tokens": 1000,
    "output_tokens": 500,
    "num_requests_per_day": 1000,
    "num_requests_per_month": 30000
  }'
```

**Response:**
```json
{
  "model": "gpt-4",
  "input_tokens": 1000,
  "output_tokens": 500,
  "num_requests_per_day": 1000,
  "num_requests_per_month": 30000,
  "cost_per_request": 0.045,
  "input_cost_per_request": 0.03,
  "output_cost_per_request": 0.015,
  "margin_cost_per_request": 0.0,
  "daily_cost": 45.0,
  "daily_input_cost": 30.0,
  "daily_output_cost": 15.0,
  "daily_margin_cost": 0.0,
  "monthly_cost": 1350.0,
  "monthly_input_cost": 900.0,
  "monthly_output_cost": 450.0,
  "monthly_margin_cost": 0.0,
  "input_cost_per_token": 3e-05,
  "output_cost_per_token": 6e-05,
  "provider": "openai"
}
```

## Related Features

- [Provider Margins](/docs/proxy/provider_margins) - Add fees or margins to LLM costs
- [Provider Discounts](/docs/proxy/provider_discounts) - Apply discounts to provider costs
- [Cost Tracking](/docs/proxy/cost_tracking) - Track and monitor LLM spend

