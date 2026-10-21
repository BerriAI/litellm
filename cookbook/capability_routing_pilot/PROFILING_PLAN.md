# Capability profiling protocol

## Current phase

This phase labels the pinned public SWE-bench dev candidates with GPT-5.4-mini and drafts an availability-aware task manifest. The hard label budget is $5. The completed label run cost $0.109114 for 220 candidates. No solver, Jev, smoke, or other model calls are part of this phase

The candidate corpus is `princeton-nlp/SWE-bench` dev at revision `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`, excluding the five held-out IDs listed in the draft manifest. The original five training tasks remain eligible, but no earlier outcomes may be reused. The labeler sees only sanitized problem statements and uses the existing rubric, a 120-second timeout, no retries, and concurrency bounded at four. Label records are append-only and do not write to `costs_aux.jsonl`

Image discovery uses registry manifest requests only and does not pull images. Each supported category is walked in deterministic seed-hash order until 30 images are confirmed or its candidates are exhausted. The process records every inspected candidate and leaves the rest explicitly not inspected. It prefers HEAD before reading manifest content, paces requests at one per second, uses at most four workers, and honors bounded `Retry-After` backoff without retry loops. Only HTTP 404 is recorded as unavailable. Authentication, rate-limit, and other errors remain errors and are not selected. No local Docker credentials were present, so discovery uses the anonymous registry token flow

## Draft cohort

The deterministic selection seed is `litellm-capability-profiling-v1`. Candidates are ordered by `sha256(seed + task_id)` within each category, and the first 30 confirmed public-image tasks are selected from `repository_debugging` and `feature_implementation`. `shell_build` and `algorithms_data` are explicitly unsupported. Unknown labels are excluded from solves and reported. The manifest remains a draft until reviewed and frozen

The effective API ceiling is $185, consisting of a $5 labeling allowance and a $180 solver allowance. The solver limit is 180 runs, at most 60 tasks across three solver models. A future run uses $1 per run, at most 50 steps, and a 600-second timeout, with three workers. Keep at least 50 GiB free. Process images in batches of three and clean them up after validation and solving. Validate each small image batch with gold and noop runs before solving. Compile an intermediate capability map after each batch with `final=false`. Freeze the reviewed manifest before any solve

## Profile records and routing

Task-level records use tenant `public_dev_poc`. Profile keys include category, model alias, the concrete returned model version, deployment ID when reported, harness and budget versions, and a canonical harness-configuration hash. Infrastructure attempts contribute spend and latency but not graded evidence. Repeated graded attempts for one task and key are rejected

Profiles use Beta(1,1) posterior summaries and a 5th-percentile lower bound. At least 13 graded tasks are required before a cell can qualify. This is a necessary threshold, not a general adequacy claim. Selection has no global fallback. The map builder reads only `profiling_runs.jsonl`, never the prior `runs.jsonl`

## Resource estimates

The prior 15 recorded solver durations give a mean of 187.882 seconds and a p90 of 272.907 seconds. For 180 solver runs over three workers, the corresponding estimates are 11,272.907 seconds (about 3.13 hours) at the mean and 16,374.392 seconds (about 4.55 hours) at the p90. This excludes gold/noop validation. A 60-task cohort requires 20 gold-validation batches and 20 noop-validation batches; their duration is unestimated because the available evaluator summaries contain no timing fields

At the prior mean cost of $0.377 per run, 180 runs would cost $67.86. The solver worst case is $180. The effective worst-case API ceiling including the $5 labeling allowance is $185

The 60 selected manifests contain 86,932,156,867 compressed layer bytes in total (80.962 GiB). The largest planned three-image batch is 6,264,019,473 bytes (5.834 GiB). With 79 GiB free at discovery, sequential batches and cleanup preserve the 50 GiB reserve. No images were pulled

## Evidence boundaries

Do not modify or reuse outcomes from `runs.jsonl`, `runs_raw/`, `eval_gold/`, `labels.json`, `costs_aux.jsonl`, or `tasks.json`. Keep held-out evaluator logs and gold patches local. Do not include credentials, gold patches, evaluator output, or private logs in the cookbook artifacts. The cancelled held-out commit is provenance only and must not be executed
