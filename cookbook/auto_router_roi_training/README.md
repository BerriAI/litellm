> **Historical benchmark invalidated:** the 25-task live run exposed upstream fixes through Git objects outside the task checkout. Its quality and savings figures are withdrawn. These files retain the audit record, not evidence of routing improvements. Fresh isolated experiments are in progress

# Experimental Auto Router training snapshots

These opt-in profiles compare the original cards, a research-informed card rewrite, and cards with training-derived capability priors. They also compare raw probabilities, per-model logit calibration, and regularized task-dependent logit calibration. The router makes one judge call; all learned probability adjustments and threshold comparisons run locally

The sample proxy configuration requires a build containing this PR. It does not change the default classifier or its configuration

## Run a profile

Set `ROI_GATEWAY_BASE_URL` to your gateway's OpenAI-compatible URL ending in `/v1`, `ROI_GATEWAY_API_KEY` to your gateway key, and `ROI_LOCAL_MASTER_KEY` to a separate local proxy key. Start the proxy from the repository root

```sh
uv run litellm --config cookbook/auto_router_roi_training/proxy.yaml --port 4000
```

Send your benchmark's first task request using one of these model aliases. Pin that selected solver for the whole attempt when comparing against the task-level experiment

| Model pair | Zero observed validation loss | Two-point validation allowance | Five-point validation allowance |
|---|---|---|---|
| Sonnet 5 / Opus 5, high effort | `roi-sonnet-opus-validation-regret-0` | `roi-sonnet-opus-validation-regret-0p02` | `roi-sonnet-opus-validation-regret-0p05` |
| GPT-5.6 Luna / Sol, high effort | `roi-luna-sol-validation-regret-0` | `roi-luna-sol-validation-regret-0p02` | `roi-luna-sol-validation-regret-0p05` |

```sh
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $ROI_LOCAL_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"roi-luna-sol-validation-regret-0","messages":[{"role":"user","content":"Describe your complete coding task here"}]}'
```

`profiles.json` contains the exact cards, coefficients, thresholds and validation results for each alias. `manifest.json` records model versions, effort, sources and limitations. `training_records.jsonl` retains the numerical forecasts and paired outcome evidence used for fitting and selection

For controlled comparisons, `fixed_boundary_profiles.json` contains 18 additional configurations: each model pair, each of the three cards, and raw, per-model, or task-dependent calibration. Calibration uses the fixed middle regularization strength of 10. These keep the original routing boundary unchanged and were not selected using live outcomes. Replace one alias's `complexity_router_config` with the chosen entry's `complexity_router_config` value to benchmark it

## What was fitted

DeepSWE v1.1 supplies 113 tasks, with repeated attempts for each solver. We split repositories into 83 training tasks and 30 validation tasks. Within each task, repeated outcomes are averaged; each task receives equal fitting weight. Missing bills remain missing and are excluded as matched pairs only from cost calculations

The trained cards add overall and language-level success priors computed exclusively on the training split. Language means receive eight pseudo-tasks of shrinkage toward the overall mean. A separate research card tests qualitative changes to the capability rules and model profiles

Post-processing uses `sigmoid(intercept + slope * logit(raw_p) + matching_task_offsets)`. The slope is positive. Task offsets use the capability rule for the capability classifier, or the reasoning, scope, specification and verification labels for V2. Training minimizes logistic cross-entropy against task success rates, with regularization strengths 1, 10 and 100. The intercept is unpenalized, the slope receives one tenth of the task-offset penalty, and offsets are bounded to [-5, 5]

Cards, calibration variants and boundaries are selected on validation. Boundary candidates are rounded to four decimal places. The objective minimizes mean solver cost plus judge cost within each declared validation quality allowance. The values in an alias name are selection constraints, not guaranteed production quality losses. A tuned V2 gap such as 0.1649 is the fitted policy boundary; it does not establish a sixteen-point real-world loss

## Interpreting the result

The raw model probabilities and calibrated probabilities are separate outputs. Better average calibration does not establish better task ranking or lower routing cost. The selected configuration can retain an original card or omit calibration when those alternatives performed better

These are small-sample experimental snapshots. Model, judge, prompt, effort, task distribution and harness changes can invalidate the coefficients. Published DeepSWE outcomes use different budgets and serving configurations from the live SWE-bench pilot, so that pilot measures transfer. Zero observed validation loss is a point estimate, not statistical noninferiority

The capability classifier uses an absolute efficient-model probability threshold plus its boundary step. V2 compares the two models' predicted probabilities. Their threshold values have different meanings and should not be copied between classifiers

## Fresh benchmark results

Read [FINDINGS.md](FINDINGS.md) for the interpretation and next experiments. See [BENCHMARK.md](BENCHMARK.md) for the 25-task comparison against the original classifiers and each fixed model. [ABLATIONS.md](ABLATIONS.md) holds the boundary fixed to isolate card and calibration changes. The JSON files include per-task routes, outcomes, cost, lost and gained solves, probability diagnostics, and the frozen selection hash

The learned Sonnet/Opus V2 profiles choose Opus on every live task and add classifier cost. That configuration provides no savings on this sample. Use the complete tables to compare other profiles with both fixed-model baselines; validation gains do not establish transfer to this workload
