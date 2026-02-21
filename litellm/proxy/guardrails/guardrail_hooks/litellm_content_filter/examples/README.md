# Content Filter Examples

## Industry-specific competitor intent (airline)

Use **competitor_intent_type: airline** for simplified config; competitors are auto-loaded from IATA `major_airlines.json`, excluding your `brand_self`. Use **competitor_intent_type: generic** when you want to specify competitors manually.

- **brand_self**: Your brand names and aliases (e.g. `["your-brand", "your-code"]`)
- **competitors** (generic only): List of competitor names

To make it effective for a specific industry (e.g. airlines), add an **industry layer** on top:

1. **domain_words** – Terms that signal “this is about our vertical.”  
   For airline: `airline`, `carrier`, `flight`, `business class`, `lounge`, etc.  
   This enables the **category_ranking** path (e.g. “Which Gulf airline is the best?”) and the scoring **gate** (so “best” alone doesn’t trigger without domain/geo).

2. **route_geo_cues** – Optional geography/hub terms.  
   For airline: `country`, `hub-city`, `airport-code`, `region`.

3. **descriptor_lexicon** – Phrases that count as indirect competitor reference.  
   For aviation: `gulf carrier`, `five star airline`, etc.

4. **competitor_aliases** – Per-competitor aliases (IATA codes, nicknames).  
   Example: `competitor-name` → `["iata-code", "nickname"]`.

5. **policy** – What to do per intent band: `refuse`, `reframe`, `log_only`, or `allow`.  
   Example: `competitor_comparison: refuse`, `category_ranking: reframe`.

See the config examples below for how to add this to your proxy `guardrails` config.

### Using the example in your proxy config

In `config.yaml`:

```yaml
guardrails:
  - guardrail_name: "airline-competitor-intent"
    litellm_params:
      guardrail: litellm_content_filter
      mode: pre_call
      competitor_intent_config:
        competitor_intent_type: airline
        brand_self: [your-brand, your-code]
        locations: [relevant-country, hub-city]
        policy:
          competitor_comparison: refuse
          possible_competitor_comparison: reframe
```

Then attach this guardrail to your router/policy (e.g. `guardrails.add: [airline-competitor-intent]`).

### Other industries

Use the same pattern:

- **SaaS**: `domain_words`: `["platform", "tool", "solution", "integration"]`; optional `route_geo_cues` if regional.
- **Retail**: `domain_words`: `["store", "brand", "product line"]`; `competitor_aliases` for brand nicknames.

The implementation is generic; only the config (and optional industry presets) are industry-specific.
