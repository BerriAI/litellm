# 05 Budgets, rate limits, and spend tracking

How much each identity may spend and how fast, plus how spend is recorded and reported: budget objects and windows, TPM/RPM and parallel request limits, the `/spend/*` and `/global/spend/*` reporting endpoints, tags, and billing exports. The identities these attach to live in [04-auth-identity.md](04-auth-identity.md); model-level provider budgets live in [03-routing-reliability.md](03-routing-reliability.md) (`routing.provider_budget_routing`)

## Budgets

### spend.budgets.key: Key budgets (max_budget, budget_duration, reset)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/virtual_keys, https://docs.litellm.ai/docs/proxy/budget_reset_and_tz
code: `litellm/proxy/management_endpoints/key_management_endpoints.py` (`max_budget`, `budget_duration` on `/key/generate`, `/key/update`), `litellm/proxy/auth/auth_checks.py` (budget enforcement)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/quota_management/budgets/`
registry: quota_management.yaml quota_management.budget.key.*
verify: POST /key/generate {"max_budget":0.001,"budget_duration":"1d"}, make calls past the cap, and confirm 429 budget-exceeded

### spend.budgets.team: Team budgets and per-member budgets
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/team_budgets
code: `litellm/proxy/management_endpoints/team_endpoints.py` (`max_budget` on `/team/new`, `/team/{team_id}/member/{user_id}/reset_budget`), `litellm/proxy/management_endpoints/team_admin_field_permissions.py`
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/quota_management/budgets/`
registry: quota_management.yaml quota_management.budget.team.*, quota_management.budget.team_member.*
verify: create a team with max_budget, generate a team key, overspend, and confirm subsequent calls 429

### spend.budgets.user: User budgets (internal users, end users)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/users, https://docs.litellm.ai/docs/proxy/customers
code: `litellm/proxy/management_endpoints/internal_user_endpoints.py`, `litellm/proxy/management_endpoints/customer_endpoints.py` (`max_budget` fields)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: quota_management.yaml quota_management.budget.internal_user.*
verify: set max_budget on an internal user, spend past it via that user's keys, and confirm the block

### spend.budgets.org: Organization budgets
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/user_management_heirarchy
code: `litellm/proxy/management_endpoints/organization_endpoints.py` (`max_budget` on `/organization/new`, `/organization/update`)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: quota_management.yaml quota_management.budget.organization.*
verify: give an org a small budget, drive team keys under it past the cap, and confirm denial

### spend.budgets.tag: Tag budgets
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/tag_budgets, https://docs.litellm.ai/docs/proxy/request_tags
code: `litellm/proxy/management_endpoints/tag_management_endpoints.py` (`/tag/*`), `litellm/proxy/spend_tracking/spend_management_endpoints.py` (GET `/spend/tags`)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.tag.*
verify: send calls with `x-litellm-tags: ["proj"]`, set a budget on tag `proj` in the UI, and confirm enforcement trips

### spend.budgets.model_access_group: Model access group budgets
surfaces: api | flags: db
docs: https://docs.litellm.ai/docs/proxy/model_access_group_budgets
code: `litellm/proxy/management_endpoints/model_access_group_management_endpoints.py` (`/access_group/{access_group}/budget`)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/quota_management/budgets/`
registry: quota_management.yaml quota_management.budget.model_access_group.*
verify: PUT /access_group/{name}/budget with a cap, exhaust it through a key scoped to the group, and confirm 429

### spend.budgets.provider: Provider budgets
surfaces: config | flags: redis
docs: https://docs.litellm.ai/docs/proxy/provider_budget_routing
code: `litellm/router.py` (provider_budget_config), `litellm/proxy/spend_tracking/spend_management_endpoints.py` (GET `/provider/budgets`)
tests: `tests/test_litellm/router_utils/`
registry: none
verify: set provider_budget_config per provider, exceed it, and confirm requests reroute to the next provider

### spend.budgets.soft_and_alerts: Soft budgets and budget alerts
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/alerting, https://docs.litellm.ai/docs/proxy/ui_team_soft_budget_alerts
code: `litellm/proxy/management_endpoints/key_management_endpoints.py` (`soft_budget`), `litellm/integrations/SlackAlerting/` (alert dispatch)
tests: `tests/test_litellm/integrations/` (alerting files)
registry: none
verify: set `soft_budget` on a key, cross it, and confirm a Slack/webhook alert fires while calls still succeed

### spend.budgets.temporary_increase: Temporary budget increase
surfaces: api | flags: db
docs: https://docs.litellm.ai/docs/proxy/temporary_budget_increase
code: `litellm/proxy/management_endpoints/key_management_endpoints.py`, `litellm/proxy/management_endpoints/team_endpoints.py` (temporary budget fields)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: none
verify: raise a key budget temporarily per the doc, confirm calls resume, and confirm the original cap returns after expiry

### spend.budgets.reset_tz: Budget reset windows and timezones
surfaces: config | flags: db
docs: https://docs.litellm.ai/docs/proxy/budget_reset_and_tz
code: `litellm/proxy/_types.py` (budget reset fields), `litellm/proxy/auth/auth_checks.py`
tests: `tests/test_litellm/proxy/` (budget reset files)
registry: quota_management.yaml quota_management.budget.key_multi_window.*
verify: set a budget_duration and timezone on a budget object, wait for (or simulate) window rollover, and confirm spend resets

### spend.budgets.budget_objects: Reusable budget objects (/budget/*)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/virtual_keys
code: `litellm/proxy/management_endpoints/budget_management_endpoints.py` (`/budget/new`, `/budget/update`, `/budget/list`, `/budget/info`, `/budget/settings`, `/budget/delete`)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/management/`
registry: mgmt.yaml mgmt.budget.*
verify: POST /budget/new {"budget_id":"b1","max_budget":10,"budget_duration":"30d"}, attach `budget_id` to a key, and confirm enforcement

## Rate limits

### spend.ratelimits.tpm_rpm: TPM/RPM limits per key, team, user, model
surfaces: api, ui, config | flags: redis
docs: https://docs.litellm.ai/docs/proxy/users, https://docs.litellm.ai/docs/proxy/virtual_keys
code: `litellm/proxy/hooks/parallel_request_limiter.py`, `litellm/proxy/management_endpoints/key_management_endpoints.py` (`tpm_limit`, `rpm_limit` fields)
tests: `tests/test_litellm/proxy/hooks/`, `tests/e2e/quota_management/ratelimit/`
registry: quota_management.yaml quota_management.ratelimit.*
verify: set `rpm_limit: 2` on a key, fire 3 calls in a second, and confirm the third 429s

### spend.ratelimits.io_token: Input/output token rate limits
surfaces: api | flags: redis
docs: https://docs.litellm.ai/docs/proxy/io_token_rate_limits
code: `litellm/proxy/hooks/parallel_request_limiter.py`, `litellm/proxy/management_endpoints/key_management_endpoints.py`
tests: `tests/test_litellm/proxy/hooks/`, `tests/e2e/quota_management/ratelimit/`
registry: quota_management.yaml quota_management.ratelimit.*
verify: set input/output token limits on a key per the doc, send a large prompt, and confirm the limit trips

### spend.ratelimits.tiers: Rate limit tiers
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/rate_limit_tiers
code: `litellm/proxy/_types.py` (rate limit tier fields), `litellm/proxy/hooks/`
tests: `tests/test_litellm/proxy/`
registry: none
verify: define tiers in general_settings, tag a key with a tier, and confirm its limits differ from the default tier

### spend.ratelimits.dynamic: Dynamic rate limiting (fair share)
surfaces: config | flags: ent, redis
docs: https://docs.litellm.ai/docs/proxy/dynamic_rate_limit
code: `litellm/proxy/hooks/dynamic_rate_limiter.py`, `litellm/proxy/hooks/dynamic_rate_limiter_v3.py`
tests: `tests/test_litellm/proxy/hooks/`
registry: none
verify: enable dynamic_rate_limiter as a callback, run two keys against a saturated model, and confirm the noisy key is throttled first

### spend.ratelimits.parallel_requests: Max parallel requests per key
surfaces: api | flags: redis
docs: https://docs.litellm.ai/docs/proxy/virtual_keys
code: `litellm/proxy/hooks/parallel_request_limiter.py`, `litellm/proxy/management_endpoints/key_management_endpoints.py` (`max_parallel_requests`)
tests: `tests/test_litellm/proxy/hooks/`
registry: none
verify: set `max_parallel_requests: 1` on a key, run two concurrent streaming calls, and confirm the second is rejected

### spend.ratelimits.headers: Rate limit response headers and pacing
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/response_headers
code: `litellm/proxy/common_request_processing.py`, `litellm/proxy/hooks/parallel_request_limiter.py`
tests: `tests/e2e/quota_management/ratelimit/` (pacing header coverage)
registry: quota_management.yaml quota_management.ratelimit.*
verify: curl -i a chat completion on a rate-limited key and read `x-ratelimit-remaining-*` / `retry-after` headers

## Spend tracking and reporting

### spend.tracking.spend_logs: Spend logs (/spend/logs, LiteLLM_SpendLogs, store_prompts_in_spend_logs)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/cost_tracking, https://docs.litellm.ai/docs/proxy/ui_spend_log_settings
code: `litellm/proxy/spend_tracking/spend_management_endpoints.py` (GET `/spend/logs`, `/spend/logs/ui`, `/spend/logs/ui/{request_id}`), `litellm/proxy/spend_tracking/` (spend log writer)
tests: `tests/e2e/quota_management/spend_tracking/`
registry: quota_management.yaml quota_management.spend_tracking.*
verify: run one chat completion, then GET /spend/logs?request_id=<id> and confirm the row with model, tokens, and cost

### spend.tracking.cost_calc: Response cost calculation (completion_cost, cost per endpoint)
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/cost_tracking, https://docs.litellm.ai/docs/completion/usage
code: `litellm/litellm_core_utils/llm_cost_calc/` (`completion_cost`, `cost_per_token`)
tests: `tests/test_litellm/litellm_core_utils/llm_cost_calc/`
registry: none
verify: python -c 'import litellm; print(litellm.completion_cost(completion_response=r))' after a completion, or read `x-litellm-response-cost` on a proxy call

### spend.tracking.tags: Request tags and tag management
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/request_tags, https://docs.litellm.ai/docs/tutorials/tag_management
code: `litellm/proxy/management_endpoints/tag_management_endpoints.py` (`/tag/new`, `/tag/list`, `/tag/info`, `/tag/daily/activity`), `litellm/proxy/litellm_pre_call_utils.py` (tag extraction)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.tag.*
verify: open http://localhost:4000/ui/?page=tag-management, create a tag, send a call with that tag, and see spend grouped under it

### spend.tracking.end_user_tracking: End user (customer) spend attribution
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/customer_usage, https://docs.litellm.ai/docs/proxy/customers
code: `litellm/proxy/management_endpoints/customer_endpoints.py` (`/end_user/daily/activity`), `litellm/proxy/spend_tracking/spend_management_endpoints.py` (`/spend_logs/end_users`, `/global/all_end_users`, `/global/spend/end_users`)
tests: `tests/e2e/quota_management/spend_tracking/`
registry: quota_management.yaml quota_management.spend_tracking.end_user.*
verify: send a call with `user: "cust-x"`, then GET /end_user/daily/activity and confirm spend attributed to cust-x

### spend.tracking.usage_page: Usage page (new_usage, daily activity, entity usage export)
surfaces: ui, api | flags: db
docs: https://docs.litellm.ai/docs/proxy/customer_usage
code: `litellm/proxy/management_endpoints/team_endpoints.py` (`/team/daily/activity*`), `litellm/proxy/management_endpoints/internal_user_endpoints.py` (`/user/daily/activity*`), `litellm/proxy/management_endpoints/tag_management_endpoints.py` (`/tag/daily/activity`), `ui/litellm-dashboard/src/app/(dashboard)/usage/page.tsx`
tests: `tests/e2e/management/`
registry: mgmt.yaml mgmt.team.daily_activity.*
verify: open http://localhost:4000/ui/?page=new_usage, pick a date range, and confirm usage rows match GET /team/daily/activity

### spend.tracking.global_spend: Global spend reports (/global/spend/*)
surfaces: api | flags: db
docs: https://docs.litellm.ai/docs/proxy/cost_tracking
code: `litellm/proxy/spend_tracking/spend_management_endpoints.py` (`/global/spend`, `/global/spend/keys`, `/global/spend/teams`, `/global/spend/models`, `/global/spend/report`, `/global/activity`)
tests: `tests/e2e/management/`
registry: none
verify: curl http://localhost:4000/global/spend/report?start_date=...&end_date=... -H "Authorization: Bearer sk-1234" and compare totals with /spend/logs

### spend.tracking.spend_capture_rate: Spend capture rate check
surfaces: config | flags: db
docs: https://docs.litellm.ai/docs/proxy/spend_capture_rate
code: `litellm/proxy/spend_tracking/spend_management_endpoints.py` (GET `/spend/capture_rate`)
tests: `tests/test_litellm/proxy/`
registry: none
verify: run traffic, then GET /spend/capture_rate and confirm captured vs expected spend ratio is reported

### spend.tracking.spend_logs_retention: Spend logs deletion / retention / partitioning
surfaces: config | flags: db
docs: https://docs.litellm.ai/docs/proxy/spend_logs_deletion
code: `litellm/proxy/spend_tracking/` (spend log cleanup), `litellm/proxy/_types.py` (retention settings)
tests: `tests/test_litellm/proxy/`
registry: none
verify: configure spend log retention per the doc, let the cleanup job run (or invoke it), and confirm old rows are deleted while aggregates remain

### spend.tracking.billing: Billing (Lago, OpenMeter, Stripe-style customer billing)
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/proxy/billing, https://docs.litellm.ai/docs/observability/lago, https://docs.litellm.ai/docs/observability/openmeter
code: `litellm/integrations/lago.py`, `litellm/integrations/openmeter.py`
tests: `tests/test_litellm/integrations/` (billing files)
registry: logging.yaml logging.openmeter.*
verify: add `success_callback: ["lago"]` with LAGO_API_KEY, run a call, and confirm a usage event lands in Lago

### spend.tracking.cloudzero_vantage: CloudZero / Vantage / PointFive cost exports
surfaces: ui, api, config | flags: ent
docs: https://docs.litellm.ai/docs/observability/cloudzero, https://docs.litellm.ai/docs/observability/vantage, https://docs.litellm.ai/docs/observability/pointfive
code: `litellm/proxy/spend_tracking/cloudzero_endpoints.py` (`/cloudzero/*`), `litellm/proxy/spend_tracking/vantage_endpoints.py` (`/vantage/*`), `litellm/integrations/cloudzero/`, `litellm/integrations/vantage/`
tests: `tests/test_litellm/integrations/cloudzero/`
registry: logging.yaml logging.cloudzero.*
verify: PUT /cloudzero/settings with a connection, POST /cloudzero/dry-run, and confirm the export preview shows spend rows

### spend.tracking.pricing_calculator: Pricing calculator
surfaces: ui, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/pricing_calculator
code: `litellm/proxy/spend_tracking/spend_management_endpoints.py` (POST `/spend/calculate`)
tests: `tests/test_litellm/proxy/`
registry: none
verify: POST /spend/calculate with a model and token counts, and confirm it returns the estimated cost

### spend.tracking.cost_tracking_settings: Cost tracking settings page
surfaces: ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/cost_tracking
code: `litellm/proxy/management_endpoints/cost_tracking_settings.py` (`/config/cost_*` endpoints), `ui/litellm-dashboard/src/app/(dashboard)/cost-tracking/page.tsx`
tests: `tests/test_litellm/proxy/`
registry: mgmt.yaml mgmt.cost_tracking.*
verify: open http://localhost:4000/ui/?page=cost-tracking, toggle a setting, and confirm it persists via GET on the matching /config/* endpoint
