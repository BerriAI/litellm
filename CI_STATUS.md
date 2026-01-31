# Release 2026-01-31 - CI Stabilization

**Started:** 2026-01-31 21:22 UTC  
**Last Updated:** 2026-01-31 22:42 UTC  
**Latest Pipeline:** #56691 - 43 passed, 8 failed, 2 running

---

## 📊 Current Failures (Pipeline #56691)

| Job | Status | Notes |
|-----|--------|-------|
| `search_testing` | ⚠️ Infra | Perplexity API 401 - needs CI credentials |
| `litellm_mapped_tests_core` | ⚠️ Infra | `@respx.mock` not intercepting real API calls |
| `litellm_proxy_unit_testing_part2` | 🔍 Investigating | Schema migration + route test |
| `e2e_openai_endpoints` | 🔍 Investigating | - |
| `proxy_e2e_anthropic_messages_tests` | 🔍 Investigating | Bedrock API 400 error |
| `proxy_logging_guardrails_model_info_tests` | 🔍 Investigating | - |
| `proxy_spend_accuracy_tests` | 🔍 Investigating | Budget related - but budget fix (#20191) not the cause |
| `build_and_test` | 🔍 Investigating | Budget related - but budget fix (#20191) not the cause |

---

## ✅ Fixed (PRs Merged)

| Job | Fix | PR |
|-----|-----|-----|
| `litellm_mapped_tests_litellm_core_utils` | test_string_cost_values prompt_tokens | [#20185](https://github.com/BerriAI/litellm/pull/20185) |
| `local_testing_part1` | Flaky error type assertion | [#20186](https://github.com/BerriAI/litellm/pull/20186) |

## ❌ PR Closed (Not the root cause)

| Job | PR | Reason |
|-----|-----|--------|
| Budget tests | [#20191](https://github.com/BerriAI/litellm/pull/20191) | Issue existed for a long time - not root cause of current failures |

---

## Next Steps

1. Investigate `litellm_proxy_unit_testing_part2` failures
2. Investigate `e2e_openai_endpoints` failures  
3. Investigate `proxy_e2e_anthropic_messages_tests` Bedrock 400 error
4. Investigate `proxy_logging_guardrails_model_info_tests`
5. Determine actual root cause of budget test failures

---

## PR Rules (from CI_AGENT_RULES.md)

1. **Title prefix:** MUST start with `litellm_`
2. **Regression link:** Find and link the commit that introduced the bug
3. **Minimal fix:** Smallest possible change
