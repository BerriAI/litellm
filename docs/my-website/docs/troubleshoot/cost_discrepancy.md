# Debugging a cost discrepancy

Cost discrepancies between LiteLLM and your provider bill usually come from one of three areas: token ingestion, the cost formula LiteLLM applies, or stale or incorrect pricing in the model map. This page walks through how to tell which case you are in.

## Step 1: Pick a time range

Lock down a specific window where the discrepancy is visible.

- Use at least 7 days of data when you can.
- Prefer a window with stable usage so one-off spikes do not dominate the comparison.
- Set the **same start and end time** on both your provider dashboard and the LiteLLM UI.

![LiteLLM dashboard date range picker](/img/cost-discrepancy-debug/date-range-picker.png)

## Step 2: Confirm traffic only goes through LiteLLM

If any requests hit the provider directly (bypassing LiteLLM), the provider will show higher usage. That is expected, not a LiteLLM bug.

Before continuing, confirm:

- All clients use your LiteLLM proxy base URL.
- No SDK or script uses provider API keys against the provider directly for the models you are comparing.
- During the selected period, the models in question are only called via LiteLLM.

If you are unsure, filter the provider dashboard by the API key or IAM principal LiteLLM uses, rather than comparing to your whole account.

## Step 3: Compare token categories

In the LiteLLM UI, open **Model activity** (under Usage analytics) so you can inspect spend and tokens per model.

![Navigate to Model activity in the LiteLLM UI](/img/cost-discrepancy-debug/go-to-model-activity.png)

Scroll the **Model** list and select the model you are reconciling with your provider bill.

![Scroll to your model in the Model activity table](/img/cost-discrepancy-debug/scroll-to-model.png)

With the same time range on both sides, fill in:

| Category | LiteLLM | Provider | Delta |
| --- | --- | --- | --- |
| Total requests | — | — | — |
| Input tokens | — | — | — |
| Output tokens | — | — | — |
| Cache read tokens | — | — | — |
| Cache write tokens | — | — | — |

LiteLLM surfaces per-category token usage for the selected model—for example prompt, completion, and cache-related tokens.

![LiteLLM usage breakdown by token category](/img/cost-discrepancy-debug/token-categories.png)

Compare these figures with your provider’s usage view (for example AWS billing tools, Azure Monitor, or the OpenAI usage dashboard) for the same period.

### Cache token reporting

- **OpenAI:** Cache read tokens are typically included inside the reported input token count.
- **Anthropic:** Cache read tokens are often reported separately from non-cached input tokens.

Compare the correct columns on each side so you are not treating “input” differently between dashboards.

### Why use a 10% threshold?

Provider dashboards and LiteLLM do not bucket requests on identical timestamps. A call at 11:59 PM can land in different daily totals on each side. Token counts can also differ slightly due to rounding across SDKs and APIs. A delta **under ~10%** is often explained by boundary effects and rounding. A delta **over ~10%** usually means something is miscounted, dropped, or categorized differently.

## Step 4: Follow the right path

<svg width="100%" viewBox="0 0 680 482" role="img" xmlns="http://www.w3.org/2000/svg" style={{ maxWidth: '100%', fontFamily: 'system-ui, sans-serif' }} aria-labelledby="cost-disc-flow-title">
  <title id="cost-disc-flow-title">Cost discrepancy debugging flowchart</title>
  <desc>Flowchart branching into Path A (token ingestion) or Path B which splits further into B1 (formula issue) and B2 (model map issue).</desc>
  <defs>
    <marker id="cd-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#888780" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </marker>
  </defs>

  <rect x="215" y="24" width="250" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" strokeWidth="0.5" />
  <text x="340" y="47" textAnchor="middle" dominantBaseline="central" fill="#444441" fontSize="14" fontWeight="500">Compare provider vs LiteLLM</text>

  <line x1="340" y1="68" x2="340" y2="104" stroke="#888780" strokeWidth="1.5" markerEnd="url(#cd-arrow)" />

  <rect x="175" y="104" width="330" height="56" rx="8" fill="#F1EFE8" stroke="#5F5E5A" strokeWidth="0.5" />
  <text x="340" y="126" textAnchor="middle" dominantBaseline="central" fill="#444441" fontSize="14" fontWeight="500">Any category off by &gt; 10%?</text>
  <text x="340" y="148" textAnchor="middle" dominantBaseline="central" fill="#5F5E5A" fontSize="12">requests, input, output, cache tokens</text>

  <path d="M220 132 L100 132 L100 250" fill="none" stroke="#0F6E56" strokeWidth="1.5" markerEnd="url(#cd-arrow)" />
  <text x="157" y="122" textAnchor="middle" fill="#0F6E56" fontSize="12">YES</text>

  <path d="M505 132 L580 132 L580 250" fill="none" stroke="#993C1D" strokeWidth="1.5" markerEnd="url(#cd-arrow)" />
  <text x="543" y="122" textAnchor="middle" fill="#993C1D" fontSize="12">NO</text>

  <rect x="40" y="250" width="220" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" strokeWidth="0.5" />
  <text x="150" y="271" textAnchor="middle" dominantBaseline="central" fill="#085041" fontSize="14" fontWeight="500">Path A</text>
  <text x="150" y="291" textAnchor="middle" dominantBaseline="central" fill="#0F6E56" fontSize="12">Token ingestion issue</text>

  <rect x="420" y="250" width="220" height="56" rx="8" fill="#FAECE7" stroke="#993C1D" strokeWidth="0.5" />
  <text x="530" y="271" textAnchor="middle" dominantBaseline="central" fill="#712B13" fontSize="14" fontWeight="500">Path B</text>
  <text x="530" y="291" textAnchor="middle" dominantBaseline="central" fill="#993C1D" fontSize="12">Quantities match, cost differs</text>

  <line x1="150" y1="306" x2="150" y2="370" stroke="#0F6E56" strokeWidth="1.5" markerEnd="url(#cd-arrow)" />

  <line x1="530" y1="306" x2="530" y2="318" stroke="#854F0B" strokeWidth="1.5" />
  <line x1="435" y1="318" x2="575" y2="318" stroke="#854F0B" strokeWidth="1.5" />
  <line x1="435" y1="318" x2="435" y2="370" stroke="#854F0B" strokeWidth="1.5" markerEnd="url(#cd-arrow)" />
  <line x1="575" y1="318" x2="575" y2="370" stroke="#854F0B" strokeWidth="1.5" markerEnd="url(#cd-arrow)" />
  <text x="448" y="312" textAnchor="middle" fill="#854F0B" fontSize="11">B1</text>
  <text x="562" y="312" textAnchor="middle" fill="#854F0B" fontSize="11">B2</text>

  <rect x="40" y="370" width="220" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" strokeWidth="0.5" />
  <text x="150" y="391" textAnchor="middle" dominantBaseline="central" fill="#085041" fontSize="14" fontWeight="500">Report to LiteLLM team</text>
  <text x="150" y="411" textAnchor="middle" dominantBaseline="central" fill="#0F6E56" fontSize="12">endpoints + model + screenshots</text>

  <rect x="380" y="370" width="110" height="56" rx="8" fill="#FAEEDA" stroke="#854F0B" strokeWidth="0.5" />
  <text x="435" y="391" textAnchor="middle" dominantBaseline="central" fill="#633806" fontSize="14" fontWeight="500">B1</text>
  <text x="435" y="411" textAnchor="middle" dominantBaseline="central" fill="#854F0B" fontSize="12">Fix formula</text>

  <rect x="510" y="370" width="130" height="56" rx="8" fill="#FAEEDA" stroke="#854F0B" strokeWidth="0.5" />
  <text x="575" y="391" textAnchor="middle" dominantBaseline="central" fill="#633806" fontSize="14" fontWeight="500">B2</text>
  <text x="575" y="411" textAnchor="middle" dominantBaseline="central" fill="#854F0B" fontSize="12">Fix model map</text>

  <path d="M150 426 L150 442 L340 442" fill="none" stroke="#888780" strokeWidth="0.5" strokeDasharray="4 3" />
  <path d="M340 442 L435 442 L435 428" fill="none" stroke="#888780" strokeWidth="0.5" strokeDasharray="4 3" />
  <path d="M340 442 L575 442 L575 428" fill="none" stroke="#888780" strokeWidth="0.5" strokeDasharray="4 3" />
  <text x="340" y="454" textAnchor="middle" fill="#5F5E5A" fontSize="11">if neither path resolves it,</text>
  <text x="340" y="470" textAnchor="middle" fill="#5F5E5A" fontSize="11">Open a github issue backing up with all your data</text>
</svg>

## Path A: Token quantity mismatch

If any category is off by more than about 10%, LiteLLM may not be ingesting that category correctly (or the provider dashboard is categorizing tokens differently—recheck Step 3 first).

**What to send the LiteLLM team:**

1. Screenshots of both dashboards with the date range visible.
2. Which category is off (input, output, cache reads, cache writes, or request count).
3. Endpoints used (for example `/chat/completions`, `/responses`, `/embeddings`).
4. Model names as sent in the request (for example `anthropic.claude-opus-4-5`, `gpt-4o`).

### For maintainers debugging ingestion

1. Start the proxy with verbose logging, for example:
   ```bash
   litellm --config config.yaml --detailed_debug
   ```
2. Reproduce a single request with the reported endpoint and model.
3. Inspect the raw `usage` object in each streamed chunk (if streaming) or in the final response body.
4. Compare that to the standard logging object (or the UI request log for that call).
5. Any gap between raw provider usage and what LiteLLM logs or aggregates is where ingestion may be wrong.

## Path B: Quantities match but cost is wrong

If token and request counts agree within ~10% but dollar amounts differ, focus on how cost is computed.

### B1: Formula issue

Manually compute expected cost using the provider’s token breakdown and published rates (per million tokens or per token).

Add other billed dimensions your provider applies (for example cache creation, audio, or tier surcharges). If your hand calculation matches the provider bill but not LiteLLM, the implementation in LiteLLM for that provider or modality may be wrong.

### B2: Model map issue

If the formula structure matches how the provider bills, the values in LiteLLM’s model map may be stale or incorrect. Cross-check:

- [`model_prices_and_context_window.json`](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
- The provider’s current public pricing

Inspect `input_cost_per_token`, `output_cost_per_token`, and any cache-related pricing fields for your exact model id (including provider prefix).

### For maintainers

1. Take authoritative token quantities from the user’s provider report.
2. Derive the formula that reproduces the provider’s line item.
3. Diff that against LiteLLM’s cost path for the same provider and response shape.
4. If the formula matches but numbers differ, update pricing in `model_prices_and_context_window.json` (and follow the project’s sync / backup rules for that file).
5. If the formula in code is wrong, fix the calculation and add a regression test using the user’s token breakdown.

## Still stuck?

1. Open a GitHub issue on [BerriAI/litellm](https://github.com/BerriAI/litellm) with your Step 3 comparison table, endpoints, and model names.


On the issue, it helps to clarify:

- Reproducible on demand or intermittent?
- Single model or many?
- Steady over time, or starting from a specific release date or config change?

### For LiteLLM maintainers

If Path A and Path B do not close the case after triage, **you** should reach out and **schedule a call with the customer** (support or engineering), with the Step 3 table and screenshots—before treating the issue.

## Checklist

```
□ Same time range on both dashboards
□ Confirmed no direct-to-provider traffic for those models
□ Compared: requests, input tokens, output tokens, cache tokens
□ Noted cache reporting differences (OpenAI vs Anthropic, and so on)
□ If > ~10% delta on quantities → Path A: report with screenshots, endpoints, model names
□ If quantities match → Path B: verify formula (B1) and model map pricing (B2)
□ If neither path fits → open a GitHub issue.
```

## See also

- [Spend tracking](../proxy/cost_tracking)
- [Sync model pricing from GitHub](../proxy/sync_models_github)
