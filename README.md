# Autorouter form consistency: independent browser verification

Evidence only. This branch holds no product code and is not meant to be merged.

Baseline: 2717b5f5971f701ab866e30ed91852239a8fea37 (runs from 2026-10-12, reused: same scripts, same proxy/DB setup)
Candidate: 68abe95c7b8d84c664187a43b8e7264b687dfd67 (origin/litellm_fix_autorouter_form_consistency)
68abe95 differs from 05fa4fa88518f531ee8983a28d2dd580f14db98c only in the Lite description in litellm/proxy/public_endpoints/autorouter_presets.json

## Setup and freshness

All runs were done at the candidate, which was checked out detached with a clean tree. `ui/litellm-dashboard/.next` was deleted before the dashboard started, and both processes were started at the SHA (see logs/proxy_start.txt and logs/ui_start.txt). logs/process_freshness.txt records the live listener PIDs, cwd and start times for ports 4000 and 3000 at HEAD 68abe95 with a clean tree.

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
| C2 matrix: 5 templates x Jev / Heuristics / LLM, user_turn, one saved router per cell (15) | Full 15-cell matrix not run on baseline. Representative sequences run: LLM with judge then OpenAI, Gemini, Anthropic (reset to Heuristics, judge dropped, every_request), and Heuristic v2 then 1M Context, Lite (frequency reset, Lite switched to LLM with a deepseek judge). C1 covers Jev then OpenAI | 15/15 cells pass. UI tiers equal payload tiers equal DB tiers. Classifier, deployment or judge, and frequency are unchanged in the UI. Within each family the non-tier config is byte-identical across all 5 templates |
| C2 repeat: OpenAI, then Gemini, then Anthropic, for each family | Run for LLM only, reset as above | Non-tier config is identical to the single-template cells. The payload has only the Anthropic tiers |
| Lite copy | Description claims an LLM classifier | Option text no longer mentions a classifier (screenshots/tip_68abe95/tip_lite_copy_dropdown.png) |
| C3 classifier away and back | Deployment and judge lost, Heuristic v2 came back as Rule-based | Deployment, judge and Heuristic v2 come back. Payload has only the active engine |
| C4 custom LEGAL tier, then a template | Template applied, LEGAL dropped | All 5 template options disabled (aria-disabled) with the hint "Restore built-in tiers to use a template". Draft unchanged. Restore defaults turns templates back on and Gemini applies with classifier settings kept. Start from scratch fully resets |
| C5 Capability and Fuse v2 with tiers that would be removed | No confirmation | Confirmation appears. Keep current settings changes nothing, Change method applies. Coming back keeps the judge and frequency |
| C5 adaptive only: adaptive on, models only in SIMPLE and REASONING, then Fuse v2 | not run | Confirmation appears because adaptive would be turned off. Cancel keeps adaptive on and the tiers. Accept applies, and on return adaptive is off with tiers, judge and frequency kept |
| C5 adaptive only control: same state, then Capability | not run | No confirmation, which is correct since nothing is removed. Adaptive is still on after return |

## Finding (deployment, not code)

With default settings the proxy serves templates from `LITELLM_AUTOROUTER_PRESETS_URL`, which is main's raw autorouter_presets.json, and only falls back to the bundled file on failure. A source build of 68abe95 without `LITELLM_LOCAL_AUTOROUTER_PRESETS=True` still showed the old Lite sentence, because main had not changed yet. The corrected copy reaches default deployments once the JSON lands on main. After that, every version that fetches the remote catalog shows the new text, including older releases whose Lite template still switches the classifier.

The full persisted C1 config, including opensource_classifier_config with deployment_name jev-decider, is in json/tip_68abe95/persisted_full_tip-c1-jev-openai.json (read from GET /v2/model/info, no browser rerun).

Layout: screenshots/{baseline_2717b5f,tip_68abe95}, json/ (sanitized request payloads, readback, results_*.json with the per-cell assertions), scripts/ (replay helpers), logs/

## tip_1392feb: candidate 1392feb197f52254643b9a543bcdc4fe1e23cb28 (PR 46108)

Source root /home/ubuntu/repos/litellm, detached at 1392feb with a clean tree. The `.next` directory was removed before the dashboard started. Proxy PID 21209 (cwd repo root, started 2026-10-13T00:19:08Z) and dashboard next-server PID 15818 (cwd ui/litellm-dashboard, started 2026-10-13T00:04:46Z) are recorded in logs/tip_1392feb/process_freshness.txt. Setup and env are the same as above, except the disposable Jev placeholder is `typesafe/jev-latest`. Every save is a real POST /model/new or edit, read back with GET /v2/model/info (`db_model: true`). Placeholder models only, no inference

```
uv run --no-sync python -u scripts/tip_1392feb/run_tip.py c1 c3 c4 c5 c5ad2 lite_copy c2m   # logs/tip_1392feb/run_A.log
uv run --no-sync python -u scripts/tip_1392feb/run_B.py r1 r3                               # logs/tip_1392feb/run_B_r1_r3.log
uv run --no-sync python -u scripts/tip_1392feb/run_B.py r1h                                 # same log, r1d
uv run --no-sync python -u scripts/tip_1392feb/run_B.py r2                                  # logs/tip_1392feb/run_B_r2.log
```

| Case | Result at 1392feb | Evidence |
|---|---|---|
| C1 Decisions Model + jev-decider + Every new user message, then OpenAI Family, save, readback, reopen | Pass. UI unchanged after the template. Saved as `oss_classifier`, `user_turn`, with `opensource_classifier_config.deployment_name` jev-decider. Reopen shows the same | screenshots t1392_c1a/b/c, json t1392_c1_payload, persisted_t1392-c1-jev-openai |
| C2 15 cells (5 templates x Decisions Model/Heuristics/LLM), plus OpenAI then Gemini then Anthropic per classifier | Pass, 15/15 plus 3/3 repeats. `ui_nontier_same` true for every cell. Only tiers differ. Payload and DB readback agree | t1392_c2m_*, t1392_c2r_*, results_c1_..._c2m.json |
| C3 Jev, LLM, Heuristic v2 away and back | Pass. Deployment, judge and Heuristic v2 come back. Payload holds the active engine only | t1392_c3a/b/c, t1392_c3_payload |
| C4 custom tier LEGAL, then template, restore, Start from scratch | Pass. All 5 templates disabled and the click is blocked. State and LEGAL unchanged. Restore re-enables them, Gemini applies, Start from scratch resets | t1392_c4a..d |
| C5 Capability/Fuse v2 confirm, cancel, accept, back | Pass. Dialog shown, cancel keeps the full state, accept applies, back to Complexity keeps judge and frequency | t1392_c5_Cap_*, t1392_c5_Fus_* |
| C5 adaptive only | Pass. Fuse v2 confirms, cancel keeps adaptive on, accept turns it off. Capability needs no confirmation and keeps adaptive | t1392_c5ao_* |
| Lite copy | Pass (bundled presets) | t1392_lite_copy_* |
| R1 first Complexity LLM selection from a saved and reloaded Capability: direct (r1a), via Decisions Model (r1c), via Heuristics (r1d) | Pass. Saved with `classification_rubric` omitted, reloaded in Edit, switched to Complexity, saved. Readback is `llm` with `classification_rubric: "agentic"` in all three. Via Decisions Model and Heuristics, LLM restores the remembered Capability first | t1392_r1a/c/d_*, json t1392_r1*_payload, results_r1_r3.json, results_r1h.json |
| R2 saved Complexity LLM with the rubric omitted, then Capability, then back, then save | Pass. Readback still omits `classification_rubric` | t1392_r2_implicit_roundtrip, results_r2.json |
| R3 Decisions Model with Fuse v2 history plus custom tier LEGAL, then LLM, then forced clicks on disabled Heuristics and Fuse v2 | Pass. Heuristics radio is disabled, Capability and Fuse v2 items are disabled. LLM comes back as Complexity, not the remembered Fuse v2. Forced clicks change nothing, LEGAL kept. UI state only: not saved, because the empty LEGAL tier correctly blocks Save | t1392_r3a, t1392_r3b, results_r1_r3.json |

### Bugbot 4253429194 (saved Capability/Fuse reloaded, then Complexity, pins the omitted rubric to agentic)

Observed in R1: a reloaded Capability router whose saved `classifier_llm_config` has no rubric gets `"agentic"` when switched to Complexity and saved. Classification: unchanged baseline behavior, not a regression. At 2717b5f, `classifier_type_transition.ts:21-23` sets `startsLlmRubric` whenever the transition leaves a forecast classifier for a non-forecast one, so baseline also writes agentic. At 1392feb (`:21-26`) it initializes only when no Complexity draft exists in the current form state. A reloaded forecast router has no persisted Complexity draft because only the active config is saved, which is the approved scope. The in-form case where a Complexity draft exists keeps the omission (R2). Per the reviewer's scope ruling the baseline browser rerun was not done. This comparison is code only

### Gaps (labeled, not run live)

- Fuse v2 saved and reloaded, then Complexity: not run. The harness cannot produce a savable Fuse v2 config, which needs solver profiles, a harness and a quality gap. In-form Fuse v2 to Complexity is covered by C5
- Exhausted allowances (Bugbot 4253359172) not restored: not run live, because a licensed proxy reports unlimited allowances. The author reports seven failing-before/passing-after eligibility regressions in the local and CI suite. Those were not run independently here
- No baseline browser rerun at this tip. The baseline_2717b5f evidence above is reused

## Gaps closure at 1392feb (2026-10-13)

This section closes the two live gaps left in tip_1392feb. Source `/home/ubuntu/repos/litellm` at 1392feb197f52254643b9a543bcdc4fe1e23cb28 with a clean tree, `.next` removed before the dashboard started. Proxy PID 3873 (started 2026-10-13T01:54:31Z) and dashboard PID 4381 (started 01:54:42Z) are in `logs/gaps_1392feb/process_freshness.txt`. Fresh disposable DB `litellm_qa_ar`, 14 placeholder models plus `jev-decider` (`typesafe/jev-latest`, `mode: evaluation`), see `logs/gaps_1392feb/config_sanitized.yaml`. No inference. The proxy runs with no license (`LITELLM_LICENSE=` explicitly empty), so `POST /auto_router/availability` reports `limit: 1` for every gated capability (`logs/gaps_1392feb/availability_probe_no_license.json`). Every save is a real `POST /model/new` or `PATCH /model/{id}/update`, read back from `GET /v2/model/info` with `db_model: true`. Availability responses are the real browser responses, recorded in each results file under `availability_responses`. Nothing was mocked and no product source was changed

Commands, run from the repo root with the env above: `uv run --no-sync litellm --config <disposable> --port 4000 --num_workers 1`, `LENS_DEV_PROXY_URL=http://localhost:4000 npm run dev -- -p 3000` in `ui/litellm-dashboard`, then `uv run --no-sync python scripts/gaps_1392feb/run_G.py g1`, `... run_G.py g2 g3` (G2 result in `results_g2_g3_first_run.json`) and `... run_G.py g3` (final G3 in `results_g3.json`)

| Case | Steps | Measured result | Verdict |
|---|---|---|---|
| G1a saved Fuse v2, reload, Complexity directly | LLM, judge claude-fable-5-1, Fuse v2, Efficient solver profile "Short reasoning budget", Capable solver profile "Larger reasoning budget", Harness and budget "Shell and test runner, one attempt", Maximum quality gap 0.05, Add Auto Router, reopen, Routing approach Complexity, Save | Created `llm_v2` with the four `llm_v2_config` fields, `adaptive: false`, tiers SIMPLE/REASONING, no rubric. Reopen shows Fuse v2 and all four fields. PATCH and readback: `classifier_type: llm`, `classification_rubric: "agentic"` | Pass |
| G1b same, via Heuristics | Reopen, Heuristics, LLM (Fuse v2 comes back, it is this router's own allowance), Complexity, Save | `llm`, rubric `"agentic"` | Pass |
| G1c same, via Decisions Model | Reopen, Decisions Model, LLM, Complexity, Save | `llm`, rubric `"agentic"` | Pass |
| G2 heuristic_v2 allowance | New draft: Heuristic v2, then LLM with judge and "Every new user message". Back to Heuristics. Back to LLM, create holder router `gap-g2-holder-hv2` (heuristic_v2) with POST /model/new, change frequency so the valid draft refetches, then Heuristics. Force-click Heuristic v2. Back to LLM. Delete holder, refetch, Heuristics, pick Heuristic v2, Save | Available (browser `remaining: 1`): Heuristic v2 restored. Exhausted (browser response `remaining: 0` at 02:02:13, before the click): falls back to Rule-based, Heuristic v2 menu item disabled, forced click changes nothing, Save enabled, LLM draft keeps judge and Complexity. After removal (browser `remaining: 1`): Heuristics stays Rule-based, because the fallback is now the last remembered Heuristics method, and Heuristic v2 is enabled and selectable. Saved and read back as `heuristic_v2` | Pass |
| G3 Fuse v2 (llm_v2) allowance | New draft: LLM, judge, Fuse v2 with the four fields, "Every new user message". Heuristics, then LLM. Heuristics, create holder `gap-g3-holder-fuse` (llm_v2, G1a payload), apply OpenAI Family so the Heuristics draft is valid and refetches, wait for the browser response. LLM, force-click Fuse v2, Save. Delete holder | Available (browser `remaining: 1`): Fuse v2 restored with all four fields. Exhausted (browser response `remaining: 0` at 02:12:34, before the click): LLM opens on Complexity, judge and "Every new user message" kept, Fuse v2 menu item disabled, forced click changes nothing. Saved and read back as `llm`, rubric `"agentic"`, `user_turn`, no `llm_v2_config`. After deleting the holder, availability reports `llm_v2` `remaining: 1` again | Pass |

Stale G3 attempts, kept for the record and not counted as proof: `results_g3_*_run_*.json`, `results_g2_g3_first_run.json` (its G3 part) and `g3_*_run_err*.png`. In each one the browser never received a `remaining: 0` response before the return to LLM. The away draft (Heuristics or Decisions Model, with the MEDIUM and COMPLEX tiers that Fuse had emptied) failed validation, so `add_auto_router_tab.tsx` sent `complexity_router_config: null` and the availability query key never changed. Changing frequency while on Heuristics does not change that serialized config either. The UI restored Fuse v2 from the last `remaining: 1` data. The next refetch then showed "Fuse v2 has no available allowance. Choose another option or free an existing allowance." and Save stayed disabled, so the stale state could not be saved

Not covered here: baseline 2717b5f was not rerun for these cases, which exercise candidate-only behavior. The seven failing-before/passing-after eligibility regressions cited in tip_1392feb remain separate local/CI evidence

## Final template-only candidate 59273de (2026-10-13)

PR46108 was narrowed to template selection only. Candidate 59273de63fbd3502041141207da1a0e1c3d092b7 changes 5 files against merge base 2717b5f5971f701ab866e30ed91852239a8fea37 (`logs/final_59273de/process_freshness.txt`). Source `/home/ubuntu/repos/litellm` detached at 59273de with a clean tree and no `.next` before the dashboard started. Dashboard next-server PID 3627 (cwd `ui/litellm-dashboard`, started 2026-10-13T13:09:16Z), proxy PID 4187 (cwd repo root, started 13:09:37Z). Fresh disposable DB `litellm_qa_ar`, the same 15 placeholder models with `jev-decider` on `typesafe/jev-latest` (`logs/final_59273de/config_sanitized.yaml`), `LITELLM_LOCAL_AUTOROUTER_PRESETS=True`, no inference. Every save is a real `POST /model/new`, read back from `GET /v2/model/info` with `db_model: true`. Nothing mocked. Baseline evidence above (2717b5f) is reused because the merge base is unchanged. Capability/Fuse, quotas and classifier round trips were not rerun, per scope

Commands: `uv run --no-sync litellm --config <disposable> --port 4000 --num_workers 1`, `LENS_DEV_PROXY_URL=http://localhost:4000 npm run dev -- -p 3000`, then `uv run --no-sync python scripts/final_59273de/run_5927.py c1 c4 c2m` (results in `json/final_59273de/results_c1_c4_c2m.json`) and `scripts/final_59273de/c4open.py` for the open-menu screenshot

| Case | Steps | Measured result | Verdict |
|---|---|---|---|
| C1 | Decisions Model, jev-decider, "Every new user message", OpenAI Family, Add Auto Router, read back, reopen in Edit | UI unchanged after the template (`t5927_c1a_before`, `t5927_c1b_after_openai`). POST and readback: `classifier_type: oss_classifier`, `opensource_classifier_config.deployment_name: jev-decider`, `classification_mode: user_turn`, OpenAI tiers. Reopened dialog shows Decisions Model, jev-decider, Every new user message (`t5927_c1c_reopened`) | Pass |
| C2 matrix | 5 templates x Decisions Model (jev-decider) / Heuristics / LLM (judge claude-fable-5-1), each with "Every new user message", each saved as its own router | 15/15: family, heuristic, decision model, judge and frequency unchanged in the UI. UI tiers = payload tiers = DB tiers in every cell. Within each family the non-tier payload is byte-identical across all 5 templates | Pass |
| C2 repeat | OpenAI, Gemini, Anthropic in sequence for each family, save | Final tiers are Anthropic's, DB readback agrees, non-tier payload identical to the single-template cells for all 3 families | Pass |
| C4 | OpenAI, LLM + judge + frequency, Edit tiers, add LEGAL, try a template, Restore defaults, Gemini, Start from scratch | With LEGAL every template option is disabled with "Restore built-in tiers to use a template" (`t5927_c4e_custom_template_menu_open`), a click does nothing and LEGAL stays. Restore defaults returns to the 4 built-in tiers, and Gemini then applies with judge and frequency kept (`t5927_c4c_restored_gemini`). Start from scratch resets to Heuristics, Every request, empty tiers (`t5927_c4d_start_from_scratch`) | Pass |

Evidence caveat: in `t5927_c4e_custom_template_menu_open` the red outline around the portal-rendered option list did not render. The disabled hint text is visible on every template option
