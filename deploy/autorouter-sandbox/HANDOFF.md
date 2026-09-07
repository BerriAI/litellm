# AutoRouter v3 dev gateway: handoff

## 0. Why this exists

Goal: build AutoRouter v3, the next version of LiteLLM's auto-router (today's `auto_router/complexity_router`), and be able to iterate on it FAST by dogfooding it on your own real traffic (Claude Code, scripts, whatever), without touching prod or the shared sandbox

The problem this solves: the real gateway (https://gateway.litellm-sandbox.ai, LiteLLM on EKS in us-west-1 with ~920 models stored in Postgres) is shared. Every router experiment there risks other people's traffic, requires a full 15 minute image build, and you cannot easily run unreleased branch code on it. Meanwhile running LiteLLM locally means copying hundreds of provider credentials you do not have

The solution: a second, personal LiteLLM instance (https://litellm-autorouter-sandbox.onrender.com) built from YOUR branch. It has no model credentials at all. Every model it does not know about is forwarded to the sandbox gateway through the `litellm_proxy/*` provider using one sandbox virtual key. So it inherits all 920 sandbox models live and at zero risk to anything, while every line of routing code that runs is whatever you last pushed. Push -> ~2 min -> your Claude Code session is using the new router

Mental model: **the sandbox is a dumb model backend; your Render instance is the brain you are rewriting**

## 1. Current state (verified)

Live URL: https://litellm-autorouter-sandbox.onrender.com

- `GET /health/liveliness` -> `"I'm alive!"`, `GET /health/readiness` -> `{"status":"healthy","db":"Not connected"}` (no DB by design)
- `POST /v1/chat/completions`, `model: moe-router`, prompt `hi` -> served by `anthropic/claude-haiku-4-5` (SIMPLE tier)
- Same endpoint, prompt asking for a distributed Redis token-bucket implementation with Lua and multi-region failover -> served by `anthropic/claude-opus-5` (COMPLEX tier)
- `POST /v1/messages` (Anthropic Messages format, what Claude Code speaks), `model: claude-sonnet-4-5` -> normal reply. Forwarding of non-router models works
- The same image was also verified locally under Docker and under `uv run` before deploying

The model that served a request is in the response header `llm_provider-x-litellm-model-name` (the sandbox's `x-litellm-model-name` re-emitted with the `llm_provider-` prefix by the `litellm_proxy` provider). That header is your fastest routing assertion

## 2. Where the code and deploys live

| Thing | Location |
|---|---|
| Deploy source of truth | GitHub `BerriAI/litellm_autorouter_sandbox_deploy`, branch `litellm_internal_staging` (default). A fork of `BerriAI/litellm`. Render auto-deploys on every push to it |
| Staging copy of the same commits | `BerriAI/litellm`, branch `litellm_autorouter_sandbox_deploy`. Only exists because the Devin GitHub app can push there and not to the fork. Not needed going forward; delete or keep as backup |
| Runtime config | `deploy/autorouter-sandbox/proxy_config.yaml` |
| Image | `deploy/autorouter-sandbox/Dockerfile` |
| Render blueprint | `render.yaml` at repo root |
| Alternatives (unused) | `deploy/autorouter-sandbox/docker-compose.yml`, `k8s.yaml`, `.github/workflows/deploy-autorouter-sandbox.yml` |
| Router code | `litellm/router_strategy/complexity_router/` and hooks in `litellm/router.py` |
| Router tests | `tests/test_litellm/router_strategy/test_complexity_router.py`, `test_complexity_tier_predictor.py` |
| Router evals | `litellm/router_strategy/complexity_router/evals/eval_complexity_router.py` (29 labeled cases, heuristic classifier only) |
| Router docs | `litellm/router_strategy/complexity_router/README.md` (457 lines, current and accurate) |

Branch is based on `BerriAI/litellm` `litellm_internal_staging` at `11a0c0abf0` (2026-09-05). Keep pulling that branch (`git pull https://github.com/BerriAI/litellm.git litellm_internal_staging`) so the source overlay keeps matching the base image's installed dependencies (see 4.2)

### 2.1 The iterate loop

```bash
git clone https://github.com/BerriAI/litellm_autorouter_sandbox_deploy.git && cd litellm_autorouter_sandbox_deploy
# edit litellm/router_strategy/complexity_router/*.py and/or deploy/autorouter-sandbox/proxy_config.yaml
git commit -am "feat(router): ..." && git push origin litellm_internal_staging
# ~2 min: Render dashboard > litellm-autorouter-sandbox > Events shows "Deploy live"
curl -si https://litellm-autorouter-sandbox.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"moe-router","messages":[{"role":"user","content":"hi"}]}' | grep -E "llm_provider-x-litellm-model-name|\"content\""
```

The branch has no protection rules; force push is allowed. Render "Manual Deploy > Clear build cache & deploy" if a build looks stale

### 2.2 Local loop (faster than Render for unit-level work)

```bash
export UPSTREAM_LITELLM_BASE_URL=https://gateway.litellm-sandbox.ai UPSTREAM_LITELLM_API_KEY=sk-<sandbox key> LITELLM_MASTER_KEY=sk-dev
uv run --no-sync litellm --config deploy/autorouter-sandbox/proxy_config.yaml --port 4010 --detailed_debug
# unit tests for the router
LITELLM_LOCAL_MODEL_COST_MAP=True uv run --no-sync pytest tests/test_litellm/router_strategy/test_complexity_router.py -q
# heuristic-only eval set
uv run --no-sync python -m litellm.router_strategy.complexity_router.evals.eval_complexity_router
# lint (the repo has strict gates; run before pushing anything you intend to upstream)
make lint
```

Docker parity check: `docker build -f deploy/autorouter-sandbox/Dockerfile -t ar:local . && docker run --rm -p 4000:4000 -e UPSTREAM_LITELLM_BASE_URL -e UPSTREAM_LITELLM_API_KEY -e LITELLM_MASTER_KEY ar:local`

## 3. How a request flows

```
Claude Code / SDK
  -> https://litellm-autorouter-sandbox.onrender.com   (auth: Render LITELLM_MASTER_KEY)
     LiteLLM proxy running THIS BRANCH
       model == "moe-router"?
         yes -> ComplexityRouter.async_pre_routing_hook (litellm/router_strategy/complexity_router/complexity_router.py:3226)
                 -> classify (LLM classifier call to openai/gpt-4o-mini, itself forwarded upstream via "*")
                 -> pick tier model, e.g. anthropic/claude-opus-5
         no  -> model name used as-is
       model matches "*" -> provider litellm_proxy, api_base=UPSTREAM_LITELLM_BASE_URL, key=UPSTREAM_LITELLM_API_KEY
  -> https://gateway.litellm-sandbox.ai/v1/chat/completions   (real provider creds live here)
  -> Anthropic / OpenAI / Bedrock / ...
```

Facts that follow from this:
- Any sandbox model name works unchanged from your instance. Only `moe-router` (and any other router you define) runs branch code
- The classifier call is a normal upstream request; it costs money on your sandbox key and adds its latency (timeout 3000 ms configured, then falls back to the heuristic scorer via the circuit breaker)
- All spend from everyone using your URL lands on the single sandbox virtual key, visible in the sandbox UI, not on your instance
- Streaming, tool calls, images, Anthropic `/v1/messages`, Responses API are all plain forwards; nothing special was done and nothing special is needed
- `drop_params: true` in the config silently drops params the upstream rejects; set it to false when debugging parameter issues
- Extra hop cost ~50-100 ms per request (Render Oregon -> AWS us-west-1)

## 4. The deploy artifacts, in detail

### 4.1 `proxy_config.yaml`

```yaml
model_list:
  - model_name: "*"                         # catch-all -> sandbox
    litellm_params:
      model: litellm_proxy/*
      api_base: os.environ/UPSTREAM_LITELLM_BASE_URL
      api_key: os.environ/UPSTREAM_LITELLM_API_KEY
  - model_name: moe-router                  # the router under development
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_default_model: anthropic/claude-sonnet-5
      complexity_router_config:
        classifier_type: llm
        classifier_llm_config: {model: openai/gpt-4o-mini, timeout_ms: 3000, classification_rubric: agentic}
        classifier_context_window_size: 35
        classifier_context_per_turn_chars: 500
        deployment_affinity: true
        session_affinity: false
        escalation_keywords: [LITELLM ESCALATE]
        tiers:
          SIMPLE: [anthropic/claude-haiku-4-5]
          MEDIUM: [anthropic/claude-sonnet-5]
          COMPLEX: [anthropic/claude-opus-5]
          REASONING: [anthropic/claude-fable-5]
general_settings: {master_key: os.environ/LITELLM_MASTER_KEY}
litellm_settings: {drop_params: true, telemetry: false}
```

Named `proxy_config.yaml` because the repo `.gitignore` (line 97) ignores every `config.yaml`; the first deploy failed on exactly that. Do not rename it back

Tier values can be any sandbox model name, or objects with per-tier param overrides (`{model_name: anthropic/claude-opus-5, litellm_params: {reasoning_effort: high}}`), see README "Configuration". Add more routers by adding more `model_list` entries; no DB or API call is involved on this instance

### 4.2 `Dockerfile` (overlay image)

`FROM ghcr.io/berriai/litellm:main-latest`, then `COPY` of `litellm/`, `enterprise/litellm_enterprise/`, `litellm-proxy-extras/litellm_proxy_extras/`, `schema.prisma`, `proxy_config.yaml` over `/app/.venv/lib/python3.13/site-packages`. `CMD ["--config","/app/config.yaml","--port","4000"]` goes to the base image's entrypoint `docker/prod_entrypoint.sh`, which `exec litellm "$@"`

Why: the root `Dockerfile` rebuilds dependencies and the Next.js admin UI (~15 min). The overlay builds in seconds. Trade-offs you must know:
- Adding a new Python dependency breaks it (ImportError at boot). Fix: build the root `Dockerfile` once (locally or in CI), push to a registry, set `BASE_IMAGE` to that; or vendor the dependency
- If upstream bumps the Python minor version in the base image, update `SITE_PACKAGES`
- The base image is `main-latest` and the source is your branch; if they drift far apart, pull upstream

### 4.3 `render.yaml`

Docker runtime, `plan: standard` (2 GB; `starter` at 512 MB OOMs on LiteLLM), region `oregon`, health check `/health/liveliness`, `autoDeploy: true`. Env vars: `PORT=4000`, `UPSTREAM_LITELLM_BASE_URL=https://gateway.litellm-sandbox.ai`, `UPSTREAM_LITELLM_API_KEY` (`sync: false`, entered once in the dashboard), `LITELLM_MASTER_KEY` (`generateValue: true`, Render generated it; read it under Environment). No `branch:` key, Render uses the branch chosen when the blueprint was applied. This file replaced upstream's stock `render.yaml`; fine on this branch, do not upstream it

### 4.4 Unused alternatives

`docker-compose.yml` + `.env.example` run the same image on any box. `k8s.yaml` + `.github/workflows/deploy-autorouter-sandbox.yml` build to GHCR and roll a k8s Deployment if `AUTOROUTER_SANDBOX_KUBECONFIG_B64` is set; the workflow triggers on branch `litellm_autorouter_sandbox_deploy`, so it would need retargeting to `litellm_internal_staging` to run in the fork. Delete both if Render stays the target

## 5. The router you are replacing: how v2 (`complexity_router`) works today

Read `litellm/router_strategy/complexity_router/README.md` in full; it is accurate. Summary of what matters for v3 design:

Entry point: `ComplexityRouter` (a `CustomLogger`, `complexity_router.py:1054`), wired by `Router._is_complexity_router_deployment` / `init_complexity_router_deployment` (`litellm/router.py:8789` and `:8822`), invoked per request through `async_pre_routing_hook` (`:3226`) -> `_classify_and_route` (`:3431`). Config is the pydantic model in `config.py` (1635 lines); every YAML key under `complexity_router_config` is a field there, add new knobs there first

Classifier types (`config.py:716`): `heuristic` (7 weighted keyword/length dimensions -> score -> `tier_boundaries`), `heuristic_v2` (calibrated success-probability model from the bundled UltraFeedback artifact `artifacts/ultrafeedback_tiers.json`, `tier_predictor.py`), `llm` (structured-output call to `classifier_llm_config.model` with a rubric from `classification_rubrics.py`; `agentic` is the coding-agent rubric currently in use), `heuristic_first` (scorer first, LLM only above `heuristic_first_max_tier`), `hybrid` (LLM only near a tier boundary), `custom` (plugin)

What the classifier sees: `_extract_current_ask_and_system_prompt` (`:536`) pulls the newest human ask, strips Claude Code system-reminder blocks (`_strip_reminder_blocks`, `:416`), plus up to `classifier_context_window_size` prior turns truncated to `classifier_context_per_turn_chars` each (`_extract_prior_turns`, `:771`). Images can be forwarded to the classifier (`_classifier_image_parts`, `:1665`)

Post-classification adjustments, in order: keyword tier rules and semantic overrides (`_resolve_keyword_tier_override` `:3109`, `_semantic_tier_override` `:3059`), escalation keywords (`_escalate_tier` `:2601`, `LITELLM ESCALATE` bumps one tier), stall escalation (`stall_detector.py`, repeated/erroring tool calls bump one tier), plan-mode floors, session affinity (pin model per `x-litellm-session-id`, Redis or in-memory), deployment affinity, context-window placement (`_context_window_placement` `:2501`, moves up a tier if the prompt will not fit), modality gate (`_gate_response_modality` `:2689`, text-only model + image -> higher tier), health gate (`_gate_response_health` `:2868`). Then `_pick_model_for_tier` (`:2087`) chooses within the tier pool

Observability: every decision is a `StandardLoggingRoutingDecision` (`litellm/proxy/_types.py:3730`) in `metadata.routing_decision` with `tier`, `cause` (`llm_classifier`, `heuristic_first_short_circuit`, `modality_escalation`, ...), `signals`, probabilities, and classifier cost (`common_request_processing.py:1411`). On a DB-backed instance these land in spend logs; on this instance you only get them via `--detailed_debug` logs or by adding a response header/callback (a good first v3 chore, see 8)

Known weaknesses worth designing v3 around (from the code and README, not measured): the LLM classifier is an extra 300-1500 ms and a paid call on every uncached turn; the heuristic scorer is keyword based and English-centric; the only eval set is 29 hand-written single-turn prompts and it exercises the heuristic path only; there is no eval for multi-turn agentic traces, which is exactly what Claude Code produces; tier quality is asserted, not measured against outcomes

## 6. Testing v3 on yourself

Claude Code:

```bash
export ANTHROPIC_BASE_URL=https://litellm-autorouter-sandbox.onrender.com
export ANTHROPIC_AUTH_TOKEN=<Render LITELLM_MASTER_KEY>
export ANTHROPIC_MODEL=moe-router          # or leave unset and pick models per session
claude
```

OpenAI SDK / anything else: `base_url=https://litellm-autorouter-sandbox.onrender.com/v1`, `api_key=<master key>`

Seeing what the router did per request, today: run locally with `--detailed_debug` and grep `routing_decision`, or `LITELLM_LOG=DEBUG` in Render env and read runtime logs. A/B two router configs: define `moe-router-a` and `moe-router-b` in the YAML and switch `ANTHROPIC_MODEL`

Other people can point at your URL and keep their existing model names; only `moe-router` goes through your code. They share your master key and your sandbox spend

## 7. Credentials (no values here)

- `UPSTREAM_LITELLM_API_KEY`: sandbox virtual key with all models, created in https://gateway.litellm-sandbox.ai/ui > Virtual Keys. Stored in Render > Environment only. Rotate: delete/recreate in sandbox UI, paste into Render, redeploy.
- `LITELLM_MASTER_KEY`: generated by Render, Render > Environment. What clients use. Rotate: edit value, redeploy, update your shell exports
- Sandbox admin UI login: user `admin`, password = sandbox master key (ask whoever runs the sandbox)
- Nothing else. No cloud creds are needed; Devin's AWS creds are Bedrock-only and cannot see the EKS cluster

## 8. Limitations and how to lift them

| Limitation | Lift |
|---|---|
| No admin UI, no per-user keys, no spend logs, `routing_decision` not persisted | Add `DATABASE_URL` (free Render Postgres) in Environment; Prisma migrates on boot; UI at `/ui` with admin + master key. Never point at the sandbox DB casually: your branch's migrations run against it. If you do want it: need its `DATABASE_URL`, `LITELLM_SALT_KEY`, `LITELLM_LICENSE`, and drop the `*` entry since models become native |
| Routing decision invisible on the API response | Add a response header (e.g. `x-litellm-routing-tier`, `x-litellm-routing-cause`) from `routing_decision` in `litellm/proxy/common_request_processing.py`; cheap and makes self-testing trivial |
| Cannot add Python deps | Build the root `Dockerfile` once and point `BASE_IMAGE` at it |
| Eval set is 29 single-turn heuristic-only prompts | Build a multi-turn agentic eval from your own Claude Code sessions (capture via `--detailed_debug` or a callback), label expected tiers, run the LLM classifier path too |
| Classifier latency/cost on every turn | Candidates already in code: `heuristic_first`, `hybrid`, `session_affinity`; v3 could cache by ask-hash or classify async and apply to the next turn |
| Classifier `openai/gpt-4o-mini` goes upstream; if the sandbox key loses access, router falls to `complexity_router_default_model` | Grant model in sandbox key, or change classifier model |

## 9. Decisions made and why (so you do not relitigate them)

- Proxy-to-sandbox over sharing the sandbox DB: zero credential copying, zero schema risk, ready in minutes; cost is one hop and no UI
- Render over EKS: nobody in the session had cluster credentials; Render gave push-to-deploy with zero ops
- Overlay Dockerfile over root Dockerfile: seconds vs ~15 min; accepted the new-dependency limitation
- Fork + staging branch: the Devin GitHub app cannot create forks/repos (403) and is not installed on the fork, hence the manual `git fetch ... && git push origin FETCH_HEAD:litellm_internal_staging --force` sync step. Install the Devin app on the fork to remove it
- The sandbox itself is on EKS behind an AWS ELB (from DNS), not on Render, and its models are DB rows, so "copy its config" was never an option

## 10. Suggested first steps for v3 work

1. Add routing-decision response headers (8, row 2) so every request you make shows tier + cause without logs
2. Capture a week of your own Claude Code traffic through `moe-router` (a callback that dumps `routing_decision` + the current ask, or a Render Postgres so spend logs persist)
3. Turn that into an eval set under `evals/` with the LLM classifier path enabled; measure tier agreement and, more importantly, outcome (did the cheap tier need escalation, `LITELLM ESCALATE` or stall signals)
4. Only then change classification: that is where v3 lives, `_classify_and_route` and the classifier strategies, with `config.py` as the contract
