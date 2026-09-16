# TealTiger

Self-contained PII detection, cost governance, and tool-authorization
guardrail. No API keys, no network calls in the governance path.

## Quick Start

Add to your proxy `config.yaml`:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "tealtiger"
    litellm_params:
      guardrail: tealtiger
      mode: "pre_call"             # required for redaction to actually modify the outbound request; during_call/post_call do not rewrite the request before it's sent
      policy_mode: "ENFORCE"       # or "MONITOR" to dry-run without blocking
```

## Test it

```bash
curl -i http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}],
    "guardrails": ["tealtiger"]
  }'
```

The request goes through with the SSN redacted rather than sent to the
model. Set `policy_mode: ENFORCE` + a `pii` policy `action: BLOCK` to reject
the call outright instead.

## Policy types

| type        | action options   | key params                          |
|-------------|-------------------|--------------------------------------|
| `pii`       | `REDACT`, `BLOCK` | `patterns`: `"all"` or list of names |
| `cost`      | `ENFORCE`         | `daily_limit_usd`                    |
| `tool_auth` | `ENFORCE`         | `allowlist`, `blocklist`             |

## Known limitations (tracked for follow-up PRs)

- Cost tracking currently needs a hook into token-usage data that
  `apply_guardrail` doesn't receive directly — see the `NOTE for reviewers`
  comment in `tealtiger.py`.
- PII pattern set ships with 47 built-in patterns across 5 categories
  (government ID, financial, contact, network/device, credentials) — see
  `PATTERN_CATEGORIES` in `patterns.py`. Several are intentionally broad
  (`credit_card_generic`, `us_bank_account`) and will overlap on shared
  input; scope `patterns: [...]` explicitly rather than using `"all"` if
  that matters for your use case.
