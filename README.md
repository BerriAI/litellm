# Autorouter form consistency: independent browser verification

Evidence only. This branch holds no product code and is not meant to be merged.

Baseline: 2717b5f5971f701ab866e30ed91852239a8fea37 (runs from 2026-10-12, reused: same scripts, same proxy/DB setup)
Candidate: 68abe95c7b8d84c664187a43b8e7264b687dfd67 (origin/litellm_fix_autorouter_form_consistency)
68abe95 differs from 05fa4fa88518f531ee8983a28d2dd580f14db98c only in the Lite description in litellm/proxy/public_endpoints/autorouter_presets.json

## Setup and freshness

All runs were done at the candidate, which was checked out detached with a clean tree. `ui/litellm-dashboard/.next` was deleted before the dashboard started, and both processes were started at the SHA (see logs/proxy_start.txt and logs/ui_start.txt).

```
cd litellm && git checkout --detach 68abe95c7b8d84c664187a43b8e7264b687dfd67 && rm -rf ui/litellm-dashboard/.next
export DATABASE_URL=<disposable postgres db litellm_qa_ar> STORE_MODEL_IN_DB=True \
  LITELLM_CORS_ORIGINS=http://localhost:3000 LITELLM_CORS_ALLOW_CREDENTIALS=true LITELLM_LOCAL_AUTOROUTER_PRESETS=True
uv run --no-sync litellm --config <config with 15 placeholder models, no provider keys> --port 4000 --num_workers 1
cd ui/litellm-dashboard && LENS_DEV_PROXY_URL=http://localhost:4000 npm run dev -- -p 3000
uv run --no-sync python scripts/run_tip.py c1 c3 c4 c5 c2m      # then: lite_copy, c5ad, c5ad2
```

The browser is Chrome driven by Playwright over CDP at http://localhost:3000/models-and-endpoints, using Auto-Routers and then Add Auto Router. Nothing is mocked. Every save is a real POST /model/new, and every save is read back with GET /v2/model/info, which returned `db_model: true` each time. No inference was run. Before each screenshot, the changed controls get a red outline set through the DOM.

Tests at 68abe95: `npx vitest run src/lib/autorouter_presets.test.ts src/components/add_model/add_auto_router_tab.integration.test.tsx` gave 2 files and 245 tests passed (logs/tests_tip_68abe95.log). The full 8-file mapped set passed 675 tests at 05fa4fa, and the executable code is identical between the two SHAs (logs/tests_05fa4fa.log).

## Results

| Case | Baseline 2717b5f | Candidate 68abe95 |
|---|---|---|
| C1 Decisions Model + jev-decider + Every new user message, then OpenAI Family, then save, read back and reopen | Heuristics, deployment lost, every_request saved | UI unchanged. Saved oss_classifier with deployment_name jev-decider and user_turn. Reopened router shows the same |
| C2 matrix: 5 templates x Jev / Heuristics / LLM, user_turn, one saved router per cell (15) | Templates reset the classifier, the judge and the frequency | 15/15 cells pass. UI tiers equal payload tiers equal DB tiers. Classifier, deployment or judge, and frequency are unchanged in the UI. Within each family the non-tier config is byte-identical across all 5 templates |
| C2 repeat: OpenAI, then Gemini, then Anthropic, for each family | Reset as above | Non-tier config is identical to the single-template cells. The payload has only the Anthropic tiers |
| Lite copy | Description claims an LLM classifier | Option text no longer mentions a classifier (screenshots/tip_68abe95/tip_lite_copy_dropdown.png) |
| C3 classifier away and back | Deployment and judge lost, Heuristic v2 came back as Rule-based | Deployment, judge and Heuristic v2 come back. Payload has only the active engine |
| C4 custom LEGAL tier, then a template | Template applied, LEGAL dropped | All 5 template options disabled (aria-disabled) with the hint "Restore built-in tiers to use a template". Draft unchanged. Restore defaults turns templates back on and Gemini applies with classifier settings kept. Start from scratch fully resets |
| C5 Capability and Fuse v2 with tiers that would be removed | No confirmation | Confirmation appears. Keep current settings changes nothing, Change method applies. Coming back keeps the judge and frequency |
| C5 adaptive only: adaptive on, models only in SIMPLE and REASONING, then Fuse v2 | not run | Confirmation appears because adaptive would be turned off. Cancel keeps adaptive on and the tiers. Accept applies, and on return adaptive is off with tiers, judge and frequency kept |
| C5 adaptive only control: same state, then Capability | not run | No confirmation, which is correct since nothing is removed. Adaptive is still on after return |

## Finding (deployment, not code)

With default settings the proxy serves templates from `LITELLM_AUTOROUTER_PRESETS_URL`, which is main's raw autorouter_presets.json, and only falls back to the bundled file on failure. A source build of 68abe95 without `LITELLM_LOCAL_AUTOROUTER_PRESETS=True` still showed the old Lite sentence, because main had not changed yet. The corrected copy reaches default deployments once the JSON lands on main. After that, every version that fetches the remote catalog shows the new text, including older releases whose Lite template still switches the classifier.

Layout: screenshots/{baseline_2717b5f,tip_68abe95}, json/ (sanitized request payloads, readback, results_*.json with the per-cell assertions), scripts/ (replay helpers), logs/
