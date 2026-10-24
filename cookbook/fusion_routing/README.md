# Fusion-style dual-model routing

Cognition's [Devin Fusion](https://cognition.com/blog/devin-fusion) pairs a frontier "main" model with a cheap "sidekick" model inside one agentic session: the main model plans, interprets ambiguity, and reviews; the sidekick handles mechanical, well-specified subtasks. A classifier decides per-call which one should run, and both stay live so a mid-session switch doesn't cost a fresh cache warm-up.

This cookbook shows the pattern on LiteLLM Proxy using a plain `CustomLogger.async_pre_call_hook`. No core LiteLLM changes needed.

## How it differs from `auto_router`

LiteLLM already ships an [`auto_router`](https://docs.litellm.ai/docs/proxy/auto_routing) with a semantic router (embedding similarity against example utterances) and a [complexity router](https://docs.litellm.ai/docs/proxy/auto_routing#complexity-router) (keyword/token heuristics). Both pick one model per request from the text of the current prompt.

Fusion routing is a different signal: it's session-aware (every call in a task shares a `session_id`) and it classifies by *role in the workflow* (main vs. sidekick), not by the text's apparent complexity. A short "apply this diff" call can still be a main-model judgment call, and a long code-heavy call can still be pure sidekick work. Nothing stops you from wiring `complexity_router`'s scoring in as one signal inside your `FusionClassifier` implementation below; this cookbook just adds the session/role layer on top of whatever classifier you plug in.

## Files

- `fusion_hook.py` — `FusionRoutingHook`, a `CustomLogger` that rewrites `data["model"]` based on a pluggable `FusionClassifier`. Ships with `ToolNameFusionClassifier`, a placeholder heuristic that routes by the last tool call name.
- `config.example.yaml` — proxy config wiring the hook in via `litellm_settings.callbacks`.
- `test_fusion_hook.py` — unit tests for the routing/metadata behavior.

## Quick start

1. Define your main + sidekick deployments and register the hook in `config.yaml` (see `config.example.yaml`).
2. Write a `FusionClassifier` for your workflow. `ToolNameFusionClassifier` is a starting point — swap in whatever signal fits (tool name, a cheap LLM-as-judge call, `complexity_router` scoring, etc).
3. Start the proxy:

   ```bash
   litellm --config cookbook/fusion_routing/config.example.yaml
   ```

4. Send every call in one task through the same `litellm_session_id` so the hook can see them as one session:

   ```bash
   curl http://localhost:4000/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-1234" \
     -d '{
       "model": "fusion-main",
       "litellm_session_id": "task-42",
       "messages": [{"role": "user", "content": "read config.yaml and summarize it"}]
     }'
   ```

   The response's `model` field (and `metadata.fusion_role` in the logged request) shows which deployment actually served the call. Filter `/ui/?page=logs` or your spend export by `session_id` to see the blended main/sidekick cost for a task versus an all-frontier baseline.

## Caveat

`ToolNameFusionClassifier` is intentionally dumb — it exists to show the hook's shape, not to be a production classifier. The real payoff comes from a classifier that understands your agent's workflow (what's a plan/interpret/review step vs. a mechanical execution step), not from text-surface heuristics.
