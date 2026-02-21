# Content Filter Examples

## Industry-specific competitor intent: Emirates (airline)

The **generic** competitor intent blocker only needs:

- **brand_self**: Your brand names and aliases (e.g. `["emirates", "ek"]`)
- **competitors**: List of competitor names (e.g. `["qatar airways", "etihad"]`)

To make it effective for a specific industry (e.g. airlines), add an **industry layer** on top:

1. **domain_words** – Terms that signal “this is about our vertical.”  
   For Emirates (airline): `airline`, `carrier`, `flight`, `business class`, `lounge`, etc.  
   This enables the **category_ranking** path (e.g. “Which Gulf airline is the best?”) and the scoring **gate** (so “best” alone doesn’t trigger without domain/geo).

2. **route_geo_cues** – Optional geography/hub terms.  
   For Emirates: `doha`, `dubai`, `abu dhabi`, `gulf`, `middle east`.

3. **descriptor_lexicon** – Phrases that count as indirect competitor reference.  
   For aviation: `doha airline`, `oryx airline`, `gulf carrier`, `five star airline`, `skytrax`.

4. **competitor_aliases** – Per-competitor aliases (IATA codes, nicknames).  
   Example: `qatar airways` → `["qr", "doha airline"]`, `etihad` → `["ey"]`.

5. **policy** – What to do per intent band: `refuse`, `reframe`, `log_only`, or `allow`.  
   Example: `competitor_comparison: refuse`, `category_ranking: reframe`.

See **emirates_competitor_intent_guardrail.yaml** for a full example you can copy into your proxy `guardrails` config or merge into an existing `litellm_content_filter` guardrail.

### Using the example in your proxy config

In `config.yaml`:

```yaml
guardrails:
  - guardrail_name: "emirates-competitor-intent"
    litellm_params:
      guardrail: litellm_content_filter
      mode: pre_call
      competitor_intent_config:
        brand_self: [emirates, ek]
        competitors: [qatar airways, qatar, etihad, turkish airlines]
        domain_words: [airline, carrier, flight, business class, lounge]
        route_geo_cues: [doha, dubai, abu dhabi, gulf]
        descriptor_lexicon: [doha airline, gulf carrier, five star airline]
        competitor_aliases:
          qatar airways: [qr, doha airline]
          etihad: [ey]
        policy:
          competitor_comparison: refuse
          possible_competitor_comparison: reframe
          category_ranking: reframe
          log_only: log_only
```

Then attach this guardrail to your router/policy (e.g. `guardrails.add: [emirates-competitor-intent]`).

### Other industries

Use the same pattern:

- **SaaS**: `domain_words`: `["platform", "tool", "solution", "integration"]`; optional `route_geo_cues` if regional.
- **Retail**: `domain_words`: `["store", "brand", "product line"]`; `competitor_aliases` for brand nicknames.

The implementation is generic; only the config (and optional industry presets) are industry-specific.
