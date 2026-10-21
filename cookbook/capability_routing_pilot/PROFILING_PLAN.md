# Capability profiling protocol

## Current phase

This phase labels the pinned public SWE-bench dev candidates with GPT-5.4-mini and freezes an availability-aware task manifest. The hard label budget is $5. The completed label run cost $0.109114 for 220 candidates. No solver, Jev, smoke, gold, or noop execution was launched

The candidate corpus is `princeton-nlp/SWE-bench` dev at revision `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`, excluding the five held-out IDs listed in the draft manifest. The original five training tasks remain eligible, but no earlier outcomes may be reused. The labeler sees only sanitized problem statements and uses the existing rubric, a 120-second timeout, no retries, and concurrency bounded at four. Label records are append-only and do not write to `costs_aux.jsonl`

Image discovery used registry manifest requests only and did not pull images. It inspected 80 of 220 candidates, found 62 accessible images, and received 18 HTTP 401 responses. One fresh anonymous token scoped to each repository was retried once. The JWT access claim was empty and the HEAD still returned 401 for each. An empty claim can indicate a missing or private repository, so absence is unverified. These rows are frozen as `access_error` exclusions, not confirmed unavailable images. No `Retry-After` or `RateLimit-Remaining` header was present

## Draft cohort

The deterministic selection seed is `litellm-capability-profiling-v1`. Candidates are ordered by `sha256(seed + task_id)` within each category, and the first 30 accessible tasks are selected from `repository_debugging` and `feature_implementation`. Each has at least 30 accessible candidates. `shell_build` and `algorithms_data` are explicitly unsupported. Unknown labels are excluded from solves and reported. The frozen artifact is `profiling/profiling_manifest.json`; availability and access-error bias are recorded there

The effective API ceiling is $185, consisting of a $5 labeling allowance and a $180 solver allowance. The solver limit is 180 runs, at most 60 tasks across three solver models. A future run uses $1 per run, at most 50 steps, and a 600-second timeout, with three workers. The runner accepts `--max-batches N` to stop after a bounded number of three-task batches. For task index `i` in manifest order, model order rotates by `i % 3`: `haiku-4.5, sonnet-5, opus-5`, then `sonnet-5, opus-5, haiku-4.5`, then `opus-5, haiku-4.5, sonnet-5`. Each run record stores that task's model order

Keep at least 50 GiB free and abort before any image pull if free disk is below 60 GiB. Process images in batches of three, validate with gold and noop before solving, and never replace a task that fails preflight. Clean only task-owned image tags or IDs after a batch, then verify disk; do not run `docker system prune` or `docker image prune`. Write raw results and a `final=false` map checkpoint after each batch, with a progress file for resume. The current runner entry point only builds dry-run schedules and checkpoints; it launches no pulls, gold/noop checks, or solves

## Profile records and routing

Task-level records use tenant `public_dev_poc`. Profile keys include category, model alias, the concrete returned model version, deployment ID when reported, harness and budget versions, and a canonical harness-configuration hash. The map compiler and runner validate task membership and the manifest SHA-256. Infrastructure attempts contribute spend and latency but not graded evidence. Repeated graded attempts for one task and key are rejected

Profiles use Beta(1,1) posterior summaries and a 5th-percentile lower bound. At least 13 graded tasks are required before a cell can qualify. This is a necessary threshold, not a general adequacy claim. Selection has no global fallback. The map builder reads only `profiling_runs.jsonl`, never the prior `runs.jsonl`

## Resource estimates

The prior 15 recorded solver durations give a mean of 187.882 seconds and a p90 of 272.907 seconds. For 180 solver runs over three workers, the corresponding estimates are 11,272.907 seconds (about 3.13 hours) at the mean and 16,374.392 seconds (about 4.55 hours) at the p90. This excludes gold/noop validation. A 60-task cohort requires 20 gold-validation batches and 20 noop-validation batches; their duration is unestimated because the available evaluator summaries contain no timing fields

At the prior mean cost of $0.377 per run, 180 runs would cost $67.86. The solver worst case is $180. The effective worst-case API ceiling including the $5 labeling allowance is $185

The 60 selected manifests contain 86,932,156,867 compressed layer bytes in total (80.962 GiB). The largest planned three-image batch is 6,264,019,473 bytes (5.834 GiB). With 79 GiB free at discovery, sequential batches and cleanup preserve the 50 GiB reserve. No images were pulled

## Evidence boundaries

Do not modify or reuse outcomes from `runs.jsonl`, `runs_raw/`, `eval_gold/`, `labels.json`, `costs_aux.jsonl`, or `tasks.json`. Keep held-out evaluator logs and gold patches local. Do not include credentials, gold patches, evaluator output, or private logs in the cookbook artifacts. The cancelled held-out commit is provenance only and must not be executed
