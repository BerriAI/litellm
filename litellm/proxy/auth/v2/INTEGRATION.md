# auth_v2 live integration runbook

The auth_v2 architecture (authn chain, casbin authz, `RequestAuthContext`, the
end-user and telemetry stages) is complete and unit-tested on this branch. What
remains is wiring it into the live proxy request flow. Each step below changes the
real request path that enforces spend and rate limits, so each must be verified
against a running proxy hitting real provider APIs, not unit tests alone.

Run the proxy for verification with:

    python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload

and enable the feature in `general_settings`:

    general_settings:
      auth_version: v2

---

## 1. Telemetry on the route (OTel v2)

**State.** The helper is built: `identity_span_attributes(context)` in
`litellm/proxy/auth/v2/telemetry.py`. Today v1 seeds identity inside the auth gate
via `seed_request_identity(...)` (`litellm/integrations/otel/runtime.py:30`); v2
does not call it, by design — telemetry should read the context on the route.

**Change.** In the OTel v2 route span path, read the context and set the
attributes:

    from litellm.proxy.auth.v2 import get_auth_context, identity_span_attributes
    ctx = get_auth_context(request)            # raises if auth_v2 didn't run
    span.set_attributes(identity_span_attributes(ctx))

Use `try_get_auth_context(request)` (returns None) at any call site that also runs
under v1, so it no-ops instead of raising.

**Verify (live).** Send a request under `auth_version: v2`, then confirm the trace
span carries `litellm.user_id` / `litellm.team_id` / `litellm.auth.method` /
`litellm.end_user_id`. No unit test substitutes for seeing the exported span.

---

## 2. Budget / rate-limit parity for non-key logins  (the risky one)

**State.** Pre-call hooks already enforce budgets and limits, reading fields off
the identity — `user_api_key_dict.max_parallel_requests` / `.rpm_limit` /
`.tpm_limit` / `.max_budget` / `.spend`
(`litellm/proxy/hooks/parallel_request_limiter.py:190,845-854`,
`max_budget_limiter.py`). They run in the request flow at
`proxy_server.py:9021,9145,9293` via `proxy_logging_obj.pre_call_hook(...)`,
independent of v1/v2.

**Gap.** Virtual keys get those fields populated by `get_key_object`. The master,
JWT, and OAuth authenticators return a thin identity (no budget/limit fields), so
the hooks read `None` and enforce nothing for those logins.

**Change.** Add an enrichment stage that runs after authentication: when the
identity lacks budget/limit fields, load the user and team and populate them.

    from litellm.proxy.auth.auth_checks import get_user_object, get_team_object
    # get_user_object: auth_checks.py:1650   get_team_object: auth_checks.py:1982

Populate exactly the fields the hooks read, sourced from the user/team rows.
Dependency-inject the loaders so the stage is unit-testable without a DB.

**Why this is not done blind here.** The field mapping is subtle: a JWT principal
has no key, so "the limit" is the user limit, the team limit, or their
combination, and the hooks aggregate key/user/team in a specific way. Getting it
wrong silently under- or over-enforces a customer's spend. It is additive today
(these logins currently enforce nothing, so this cannot regress existing
behavior), but the exact mapping must be confirmed on a running proxy.

**Verify (live).**
1. Create a user with `rpm_limit: 2`. Authenticate as that user via JWT.
2. `for i in $(seq 1 5); do curl -s -o /dev/null -w "%{http_code}\n" \
     -H "Authorization: Bearer <jwt>" \
     -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}' \
     http://localhost:4000/v1/chat/completions; done`
   Expect `200 200 429 429 429`.
3. Repeat with a team-level limit and with `max_budget` to confirm each path.

---

## 3. Single budget authority

**State.** A few budget caps (team / org / global proxy spend) still execute
inside the auth gate's `common_checks` (`auth_checks.py:508+`); v2 does not call
`common_checks`, so those caps are not enforced under v2.

**Change.** Move those checks into the budget pre-call hook so one path enforces
them for both v1 and v2, then delete them from `common_checks`.

**Verify (live).** Set `litellm.max_budget` (global) below current spend and
confirm requests are blocked under `auth_version: v2`, matching v1.

---

## Order

Do (2) first — it is the gap that makes v2 unusable for JWT/master deployments —
then (1), then (3). After all three, run a full real-traffic pass with
`auth_version: v2` and compare budget/limit/telemetry behavior against v1 before
considering the feature production-eligible.
