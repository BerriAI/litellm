# Budget Code Matrix

What LiteLLM actually implements for budgets: every entity that can carry a dollar
budget, how the limit is enforced, and where in the code it happens. This is the
"what we support" reference; the companion `BUDGET_TEST_COVERAGE_MATRIX.md` maps
each row to its tests and the e2e gaps.

Over-budget surfaces as a `budget_exceeded` error (the live suite
`tests/otel_tests/test_e2e_budgeting.py` asserts `type == "budget_exceeded"`,
`code == "429"`); the underlying `BudgetExceededError` is defined in
`litellm/exceptions.py` (`status_code=400`). Enforcement runs in `common_checks()`
/ `auth_checks.py` at auth time, plus pre-call reservation in
`budget_reservation.py`.

Legend for "Enforced": **block** = request rejected; **filter** = router skips the
deployment; **alert** = notify only, request proceeds.

---

## 1. Per-entity dollar budgets

| Entity | Budget stored | Hard `max_budget` | Soft budget | Per-window | Model budget | Reset by `budget_duration` |
|--------|---------------|-------------------|-------------|------------|--------------|----------------------------|
| API key | `LiteLLM_VerificationToken` (direct cols + `budget_id` FK) | block (`_virtual_key_max_budget_check`) | alert (`_virtual_key_soft_budget_check`) + 80% alert | block (`_virtual_key_multi_budget_check`) | block (`model_max_budget_limiter.is_key_within_model_budget`) | keys reset job |
| Internal user | `LiteLLM_UserTable` (direct cols) | block (`common_checks`, only when not on a team) | - | - | via `model_max_budget` json | users reset job |
| Team | `LiteLLM_TeamTable` (direct cols) | block (`_team_max_budget_check`) | alert (`_team_soft_budget_check`) | block (`_team_multi_budget_check`) | via `model_max_budget` | teams reset job |
| Team member | `LiteLLM_TeamMembership` -> `LiteLLM_BudgetTable` | block (`_check_team_member_budget`) | - | - | - | budget-table reset job |
| End-user / customer | `LiteLLM_EndUserTable` -> `LiteLLM_BudgetTable` | block (`_check_end_user_budget`) | - | - | block (`is_end_user_within_model_budget`) | budget-table reset job |
| Organization | `LiteLLM_OrganizationTable` -> `LiteLLM_BudgetTable` | block (`_organization_max_budget_check`) | - | - | via budget-table | budget-table reset job |
| Tag | `LiteLLM_TagTable` -> `LiteLLM_BudgetTable` | block (`_tag_max_budget_check`) | - | - | via budget-table | budget-table reset job |
| Project | `LiteLLM_ProjectTable` -> `LiteLLM_BudgetTable` | block (`_project_max_budget_check`) | alert (`_project_soft_budget_check`) | - | - | budget-table reset job |
| Provider (router) | config `provider_budget_config` (in-memory) | filter (`router_strategy/budget_limiter`) | - | yes (time window) | - | window TTL |
| Global proxy | `litellm.max_budget` (config) | block (`_global_proxy_budget_check`) | - | - | - | - |

Notes / flags from the code:
- **User budget only enforced off-team**: `common_checks` skips the personal-user
  budget when the key belongs to a team (team budget governs instead).
- **Comparison operators are inconsistent**: key/user use `>=`, team/end-user main
  budget use `>`. Spend exactly at `max_budget` blocks a key but not a team.
- **Provider budgets are filter-only**: an over-budget provider is removed from
  routing; if all are over budget the router raises
  `no_deployments_with_provider_budget_routing` (not a per-entity block).
- **Enforcement timing differs by entity**: key / user / org / team-member / tag /
  model enforce off real-time reservation counters (block within ~2 calls);
  **end-user** enforcement reads `EndUserTable.spend`, which only updates on the
  `proxy_batch_write_at` flush, so it lags by that interval (verified live).

## 2. Budget mechanisms

| Mechanism | What it does | Code |
|-----------|--------------|------|
| Pre-call reservation | Estimates max request cost, atomically reserves against redis spend counters for key/team/user/end_user/tag/team_member/org before the call; blocks if a counter would exceed | `spend_tracking/budget_reservation.py` |
| Post-call reconciliation | Adjusts the reservation to the actual cost once known | `reconcile_budget_reservation` |
| Read-time enforcement | Auth-time check of current spend vs `max_budget` | `auth_checks.common_checks` + per-entity `_*_max_budget_check` |
| Soft budget / alerts | At `soft_budget` (or 80% of max) fire Slack/email alert, do not block | `_virtual_key_soft_budget_check`, `_team_soft_budget_check`, `budget_alerts` |
| Multi-window budgets | `budget_limits` list of `{budget_duration, max_budget}`; each window enforced + reset independently | `_virtual_key_multi_budget_check`, `reset_budget_windows` |
| Model-level budgets | `model_max_budget` dict (per model: `budget_limit` + `time_period`) on key/user/team/member/end_user | `hooks/model_max_budget_limiter.py` |

Per-model spend pooling (user keys vs service accounts): see `claude-usage-proxy` repo → `docs/model-max-budget-enforcement.md`.
| Reset by duration | Job zeros `spend`, recomputes `budget_reset_at = now + duration_in_seconds(budget_duration)`, invalidates redis counters | `common_utils/reset_budget_job.py`, `duration_parser.duration_in_seconds` |
| Zero-cost bypass | Models with no configured price bypass budget reservation | `budget_reservation` zero-cost path |

## 3. Budget management surface (endpoints)

| Action | Endpoint | Handler |
|--------|----------|---------|
| Create budget | `POST /budget/new` | `new_budget` |
| Update budget | `POST /budget/update` | `update_budget` |
| Budget info | `POST /budget/info` (`{"budgets": [id]}`) | `info_budget` |
| Budget settings | `GET /budget/settings` | `budget_settings` |
| List budgets | `GET /budget/list` | `list_budget` |
| Delete budget | `POST /budget/delete` (`{"id": id}`) | `delete_budget` |
| Set on key | `POST /key/generate`, `/key/update` (`max_budget`, `soft_budget`, `budget_duration`, `model_max_budget`, `budget_id`) | key mgmt |
| Set on user | `POST /user/new` (`max_budget`, `budget_duration`) | internal user |
| Set on team | `POST /team/new` (`max_budget`, `soft_budget`, `team_member_budget`) | team |
| Set on team member | `POST /team/member_add` (`max_budget_in_team`) | team |
| Set on org | `POST /organization/new` (`max_budget`, `soft_budget`, `model_max_budget`) | org |
| Set on customer | `POST /customer/new`, `/customer/update` (`max_budget`, `budget_id`) | customer |
| Set on tag | `POST /tag/new`, `/tag/update` (`max_budget`) | tag mgmt |
| Read budget+spend | `/key/info`, `/user/info`, `/team/info`, `/organization/info`, `/customer/info`, `/budget/info` | per-entity info |

Endpoint method/shape gotchas verified live: `/organization/delete` is **DELETE**
with `{"organization_ids": [id]}`; `/budget/info` takes `{"budgets": [id]}`;
`model_max_budget` entries use `{"budget_limit", "time_period"}`.

## 4. Config knobs

| Setting | Effect |
|---------|--------|
| `litellm.max_budget` | proxy-wide hard cap (global proxy budget) |
| `max_internal_user_budget` / `default_max_internal_user_budget` | default `max_budget` for internal users |
| `internal_user_budget_duration` | default reset duration for internal users |
| `max_end_user_budget` / `max_end_user_budget_id` | default budget for end-users |
| `default_team_params` | default `max_budget` / `budget_duration` / limits for teams |
| `provider_budget_config` (router) | per-provider spend caps + windows |
