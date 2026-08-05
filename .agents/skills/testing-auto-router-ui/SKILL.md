---
name: testing-auto-router-ui
description: How to runtime-test Auto Router features in the LiteLLM Admin UI (Models + Endpoints > Auto-Routers), including the Test Routing modal, saving routers, and verifying routing decisions in the Logs drawer.
---

# Testing Auto Router UI features end to end

## Bringing the stack up

1. `sudo service postgresql start`
2. `(cd litellm/proxy && uv run --no-sync prisma db push --accept-data-loss --skip-generate)`
3. Start the proxy **with `STORE_MODEL_IN_DB=True`**:
   `STORE_MODEL_IN_DB=True uv run --no-sync litellm --config litellm/proxy/dev_config.yaml --port 4000`
   Without that env var, any UI action that persists a model or auto router fails with
   `Set 'STORE_MODEL_IN_DB=True' in your env to enable this feature.` The failure only shows up at
   save time, so it can silently block half a test plan. Set it from the start.
   Avoid `--detailed_debug` for long browser sessions; it makes startup and each request noticeably slower.
4. Dashboard dev server: `source ~/.nvm/nvm.sh && nvm use 24 && cd ui/litellm-dashboard && npm run dev`
   (Node 24 is required.) Dashboard at http://localhost:3000, proxy-served UI at http://localhost:4000/ui/.
5. Master key and provider keys live in `litellm/proxy/dev_config.yaml` and `.env`; read them instead of asking.
6. Proxy startup takes 60-120s. Poll `curl -s -o /dev/null -w '%{http_code}' http://localhost:4000/health/readiness`
   until it returns 200 rather than sleeping a fixed amount.
7. The dashboard session survives a proxy restart (the UI token is validated against the master key),
   so you can restart the proxy mid-test and just navigate/reload without logging in again.

## UI path

Sidebar `Models + Endpoints` > `Auto-Routers Beta` tab > `Add Auto Router` > expand
`Detailed Configuration` to set Simple/Medium/Complex/Reasoning tiers. Tier selects are
searchable multi-selects: click, type the model group name, click the suggestion, then press
Escape to close the dropdown before moving to the next tier.

Footer buttons are `Test Routing`, `Test Connection`, `Add Auto Router`. `Test Routing` stays
disabled while the config cannot route (missing tiers, keyword rules pointing at empty tiers, or a
tier naming a model group the proxy does not have). That last case means the "no model group by that
name" warning inside the modal is generally unreachable from the UI; verify it against
`POST /auto_router/test_routing` directly instead.

## Verifying routing decisions

- The modal decision card and the Logs drawer use the same `RoutingDecisionCard`, so parity checks
  are worth doing: save the router, send a prompt through it from `Playground` (Select Model =
  router name), then open Logs > newest row and compare Tier / Decided by / Score / Routed to /
  Signals against what the modal showed.
- Heuristic baselines that were stable on this config (SIMPLE=gpt-4o-mini, MEDIUM=anthropic-haiku-4-5,
  COMPLEX=anthropic-sonnet-4-5, REASONING=anthropic-opus-4-5): `what is 2+2` -> SIMPLE, score -0.15;
  a long multi-clause design/prove prompt -> COMPLEX, score 0.45.
- Short "think step by step ..." prompts may still classify as SIMPLE because prompt length dominates
  the heuristic score. Do not assume a reasoning-flavoured phrase is enough to hit the REASONING tier;
  use a long prompt if you need a high tier.
- To prove a routing test is a dry run, check both the Auto-Routers list (no deployment created) and
  Logs/`LiteLLM_SpendLogs` (no new spend row) before making any deliberate real call.

## Forcing backend-error UI states

To see how a modal renders a backend failure without editing code: fill the form while the proxy is
up, then `pkill -f "litellm --config"` and click the send/submit button. The in-modal error path
renders (`Could not route this prompt` / `Failed to fetch`) and form state survives because it lives
in React memory. Restart the proxy afterwards.

## Known rough edge

Closing and reopening the Test Routing modal can retain the previous prompt and previous
result/error, so a stale card may look like a fresh run. When testing this modal, always clear the
textarea and confirm the card actually changed rather than trusting what is on screen.

## Devin Secrets Needed

None beyond what is already in the repo's `.env` / `litellm/proxy/dev_config.yaml`
(OpenAI and Anthropic keys, `DATABASE_URL`, master key).
