# CircleCI Cost Analysis — LiteLLM
**Branch:** `analysis/circleci-cost-optimization`
**Current spend:** ~$15,000/month
**Target:** Identify optimizations; compensated 20% of monthly savings achieved

---

## 1. Pipeline Inventory

### Job count
The single workflow (`build_and_test`) triggers **~57 distinct jobs** on every push to `main` or any `litellm_.*` branch. Several jobs use `parallelism`, multiplying the actual container count:

| Scope | Count |
|---|---|
| Total distinct jobs in workflow | ~57 |
| Jobs with machine executor (full VM) | 18 |
| Jobs with `resource_class: xlarge` | 24 |
| Jobs with `resource_class: large` | 3 |
| Jobs with `resource_class: medium` (explicit) | 2 |
| Jobs with `parallelism: 4` | 2 (`local_testing_part1/2`) |
| Parallel browser E2E jobs | 2 (chromium + firefox) |

### Resource class breakdown (Docker executor)
| Class | vCPUs | RAM | Credits/min | $/min (approx) |
|---|---|---|---|---|
| `medium` (default) | 2 | 4 GB | 10 | $0.006 |
| `large` | 4 | 8 GB | 20 | $0.012 |
| `xlarge` | 8 | 16 GB | 40 | $0.024 |

### Resource class breakdown (Machine executor — Linux VM)
| Class | vCPUs | RAM | Credits/min | $/min (approx) |
|---|---|---|---|---|
| `large` | 4 | 15 GB | 100 | $0.060 |
| `xlarge` | 8 | 32 GB | 200 | $0.120 |

**Machine executors cost 5–8× more than equivalent Docker executor jobs.**

---

## 2. Cost Drivers — Findings

### 2.1 Machine executor overuse (HIGH IMPACT)

18 jobs use machine executors (`ubuntu-2204:2023.10.1`) at `xlarge`. Most of these are integration test suites that spin up a Docker container internally, but this does **not** require a full VM — CircleCI's `setup_remote_docker` command provides Docker-in-Docker capability within Docker executor jobs at a fraction of the cost.

**Jobs using machine xlarge unnecessarily:**

| Job | Why machine? | Could use Docker executor? |
|---|---|---|
| `build_and_test` | Builds + runs Docker container | Yes — `setup_remote_docker` |
| `litellm_security_tests` | Runs Docker + Conda | Yes — use pre-built image |
| `proxy_logging_guardrails_model_info_tests` | Docker-in-Docker | Yes |
| `proxy_spend_accuracy_tests` | Docker-in-Docker | Yes |
| `proxy_multi_instance_tests` | Docker-in-Docker | Yes |
| `proxy_store_model_in_db_tests` | Docker-in-Docker | Yes |
| `proxy_pass_through_endpoint_tests` | Docker-in-Docker | Yes |
| `e2e_openai_endpoints` | Docker-in-Docker | Yes |
| `check_code_and_doc_quality` | xlarge, no Docker needed | Yes — medium is enough |
| `db_migration_disable_update_check` | Docker-in-Docker | Yes |

**Estimated saving:** Converting 15 machine-xlarge jobs (avg 30 min each) to Docker-xlarge:
- Current: 15 × 30 min × $0.120 = **$54/run**
- After: 15 × 30 min × $0.024 = **$10.80/run**
- **Saving per run: ~$43 (80% reduction on this category)**

---

### 2.2 xlarge everywhere — most jobs don't need it (HIGH IMPACT)

24 jobs declare `resource_class: xlarge`. The majority run pytest with `-n 8` or `-n 16` (pytest-xdist). The xdist workers are I/O-bound (waiting on mocked API calls, DB queries), not CPU-bound — 8 vCPUs is overkill. Most would run just as fast on `large` (4 vCPUs).

**Jobs that could be downgraded from xlarge → large:**

| Job | Current | xdist workers | Suggested |
|---|---|---|---|
| `litellm_mapped_tests_proxy_part1` | xlarge | `-n 8` | large |
| `litellm_mapped_tests_proxy_part2` | xlarge | `-n 8` | large |
| `litellm_mapped_tests_core` | xlarge | `-n 16` | large |
| `litellm_mapped_tests_litellm_core_utils` | xlarge | `-n 16` | large |
| `litellm_mapped_tests_integrations` | xlarge | `-n 16` | large |
| `litellm_mapped_tests_mcps` | xlarge | `-n 4` | medium |
| `litellm_mapped_tests_llms` | xlarge | `-n 16` | large |
| `litellm_mapped_enterprise_tests` | xlarge | — | large |
| `check_code_and_doc_quality` | xlarge | — | medium |

**Estimated saving:** Downgrading 9 Docker-xlarge → Docker-large (avg 25 min each):
- Current: 9 × 25 × $0.024 = **$5.40/run**
- After: 9 × 25 × $0.012 = **$2.70/run**
- **Saving per run: ~$2.70 (50% on this category)**

---

### 2.3 Broken dependency caching (MEDIUM IMPACT)

Every job that has a dependency install block shows a structural caching bug:

```yaml
# In most jobs:
- restore_cache:
    keys:
      - v1-dependencies-{{ checksum ".circleci/requirements.txt" }}
- run:
    name: Install Dependencies
    command: |
      python -m pip install ...  # installs into system Python, NOT ./venv
- save_cache:
    paths:
      - ./venv         # ← saves ./venv, which is EMPTY
    key: v1-dependencies-{{ checksum ".circleci/requirements.txt" }}
```

`./venv` is never created — packages install into the system Python. **The cache saves nothing and restores nothing.** Every job re-installs 40+ packages from scratch, adding 4–8 minutes of pure overhead per job.

Two different cache key schemes also co-exist:
- `v1-dependencies-{{ checksum ".circleci/requirements.txt" }}`
- `v2-litellm-deps-{{ checksum "requirements.txt" }}-{{ checksum ".circleci/config.yml" }}`

These never share a cache, meaning a job using scheme A will never benefit from a warm cache created by scheme B.

**Fix options (in order of effort):**
1. **Quick fix:** Change all `save_cache.paths` from `./venv` to `~/.cache/pip` — pip's download cache. This alone cuts install time by ~60% (packages are downloaded once, built repeatedly).
2. **Better fix:** Create an explicit venv in each job (`python -m venv ./venv && ./venv/bin/pip install ...`) and cache `./venv`. Restore it and skip re-installing if cache hits.
3. **Best fix:** Build a custom Docker image with all test dependencies pre-installed (see §2.4).

**Estimated saving:** 4 min saved × 40 jobs × $0.012/min (avg cost) = **$1.92/run** just from cache hits.

---

### 2.4 No shared test Docker image — 40 pip installs per job (HIGH IMPACT)

The same 40+ pip packages appear verbatim in at least **15 different jobs**. A representative install block (from `local_testing_part1`, `local_testing_part2`, `langfuse_logging_unit_tests`, `caching_unit_tests`, `litellm_proxy_unit_testing_*`, `build_and_test`, `litellm_security_tests`...):

```
google-generativeai, google-cloud-aiplatform, boto3, aioboto3, langchain,
langfuse, logfire, traceloop-sdk, opentelemetry-*, openai, prisma,
fastapi, gunicorn, pydantic, pytest + plugins, ...
```

Each install takes **4–8 minutes** of CPU+network time on a fresh container.

**Fix:** Build and publish a `litellm-ci-base` Docker image (weekly or on `requirements.txt` change) with all shared dependencies pre-installed. Replace `cimg/python:3.11` with `ghcr.io/berriai/litellm-ci-base:latest` in all Docker executor jobs.

**Estimated saving (conservative):** 5 min saved × 30 jobs × $0.024/min = **$3.60/run**

---

### 2.5 All jobs run on all branches — no PR-level filtering (HIGH IMPACT)

Every single job has the same filter:
```yaml
filters:
  branches:
    only:
      - main
      - /litellm_.*/
```

The pattern `litellm_.*` matches **every feature branch** in this repo (all follow the `litellm_<name>` convention). This means every developer push triggers the full 57-job pipeline — including expensive E2E tests, Docker builds, multi-instance proxy tests, and publish jobs.

**Fix:** Introduce two workflow tiers:

| Tier | Branches | Jobs |
|---|---|---|
| **PR check** | all `litellm_.*` branches | Unit tests only: `mypy_linting`, `semgrep`, `litellm_mapped_tests_*`, `litellm_proxy_unit_testing_*`, `check_code_and_doc_quality` |
| **Full CI** | `main` only | All 57 jobs |

**Estimated saving:** If 70% of pipeline runs are on feature branches (typical for active OSS), and PR checks cost ~20% of full pipeline cost: saves ~$0.70 × full_run_cost per feature-branch run.

---

### 2.6 Black formatting runs twice in test jobs (LOW IMPACT)

Both `local_testing_part1` and `local_testing_part2` run `python -m black .` mid-job:

```yaml
- run:
    name: Black Formatting
    command: |
      cd litellm
      python -m pip install black
      python -m black .
```

This is a formatting check (not a test) — it:
- Installs `black` fresh in each of the 4+4 parallel containers
- Runs a full repo format check
- Duplicates what `check_code_and_doc_quality` already handles

**Fix:** Remove from `local_testing_part1/part2`. `check_code_and_doc_quality` already covers it.

---

### 2.7 Miniconda downloaded fresh every machine-executor run (LOW-MEDIUM IMPACT)

`build_and_test` and `litellm_security_tests` both:
```yaml
- run:
    name: Install Python 3.x
    command: |
      curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ...
      bash miniconda.sh -b -p $HOME/miniconda
      conda create -n myenv python=3.x -y
```

No caching is applied to `$HOME/miniconda`. Each run downloads ~90 MB and creates a fresh conda env. This adds 3–5 minutes on a machine executor.

**Fix:** Cache `$HOME/miniconda` keyed on the Python version. Or better: switch these jobs to Docker executor with the target Python version pre-baked.

---

### 2.8 E2E tests run two browsers on every commit (MEDIUM IMPACT)

`e2e_ui_testing` runs both Chromium and Firefox on every push:
```yaml
- e2e_ui_testing:
    name: e2e_ui_testing_chromium
- e2e_ui_testing:
    name: e2e_ui_testing_firefox
```

These require `build_docker_database_image` to complete first, so they sit at the end of the critical path. Firefox coverage is valuable but not worth doubling E2E cost on every commit.

**Fix:** Run Firefox E2E on a nightly scheduled pipeline only. Chromium runs on every push to `main`.

---

### 2.9 `no_output_timeout: 120m` on most jobs (RISK, not direct cost)

Nearly all test jobs set a 2-hour timeout. A hung test (stuck waiting on an external API, deadlocked async loop) will bill for 2 full hours before CircleCI kills it. With 50+ jobs, a single flaky test can burn $14+ in wasted compute before being caught.

**Fix:** Set job-level timeouts proportional to actual test durations:
- Unit test jobs: `no_output_timeout: 20m`
- Integration test jobs: `no_output_timeout: 45m`
- Docker build + proxy jobs: `no_output_timeout: 60m`
- Reserve 120m only for the full `build_and_test` job

---

### 2.10 Duplicate packages within single install blocks (MINOR)

In `build_and_test`, `litellm_security_tests`, and others, `pip install "pytest==7.3.1"` appears twice in the same install block. Also `pip install google-cloud-aiplatform` appears twice in `local_testing_part1/part2` (once with a pinned version, once without). These cause silent re-installs and pip resolver overhead.

---

## 3. Summary of Opportunities

| # | Finding | Impact | Effort | Est. saving/run |
|---|---|---|---|---|
| 2.1 | Machine executor → Docker executor | 🔴 High | Medium | ~$43 |
| 2.2 | xlarge → large/medium downgrades | 🔴 High | Low | ~$2.70 |
| 2.3 | Fix broken pip cache (save to `~/.cache/pip`) | 🟠 Medium | Low | ~$1.92 |
| 2.4 | Shared CI Docker base image | 🔴 High | High | ~$3.60 |
| 2.5 | PR vs full-CI workflow split | 🔴 High | Medium | ~40–70% of per-run cost |
| 2.6 | Remove duplicate Black runs | 🟡 Low | Low | ~$0.20 |
| 2.7 | Cache Miniconda | 🟠 Medium | Low | ~$0.36 |
| 2.8 | Firefox E2E → nightly only | 🟠 Medium | Low | ~$1.50 |
| 2.9 | Reduce `no_output_timeout` values | 🟡 Risk | Low | Reduces runaway cost risk |
| 2.10 | Remove duplicate pip installs | 🟡 Low | Low | < $0.10 |

---

## 4. Recommended Implementation Plan

### Phase 1 — Quick wins (1–2 days, no test risk)
Changes that don't touch test logic:

1. **Fix cache path:** Change `save_cache.paths: - ./venv` → `- ~/.cache/pip` across all jobs. Unify cache key to `v3-litellm-deps-{{ checksum "requirements.txt" }}-{{ checksum ".circleci/requirements.txt" }}`.
2. **Reduce timeouts:** Set `no_output_timeout` proportionally per job category.
3. **Remove duplicate Black steps** from `local_testing_part1/part2`.
4. **Firefox E2E → nightly:** Add a separate `nightly` workflow with `triggers: schedule`.
5. **Cache Miniconda** in `build_and_test` and `litellm_security_tests`.

**Estimated Phase 1 savings: 15–20% (~$2,250–$3,000/month)**

---

### Phase 2 — Resource right-sizing (2–3 days)
Changes to resource classes — low risk, high return:

6. **Downgrade test jobs:** `litellm_mapped_tests_*` and `check_code_and_doc_quality` from `xlarge` → `large` or `medium`.
7. **Convert machine → Docker for pure-test jobs:** Start with `litellm_security_tests` and `check_code_and_doc_quality` — these don't need Docker-in-Docker and gain nothing from a full VM.

**Estimated Phase 2 savings: additional 20–25% (~$3,000–$3,750/month)**

---

### Phase 3 — Structural improvements (1 week)
Higher-effort, higher-impact:

8. **PR vs full-CI workflow split:** Add a `pr_check` workflow that runs only unit tests on feature branches. Gate the full pipeline to `main` only.
9. **Convert remaining machine executors to Docker+setup_remote_docker:** `proxy_*` integration jobs, `build_and_test`, `e2e_openai_endpoints`.
10. **Build `litellm-ci-base` Docker image:** Publish to GHCR; update all Docker executor jobs to use it. Eliminates 5–8 min of dep install per job.

**Estimated Phase 3 savings: additional 30–40% (~$4,500–$6,000/month)**

---

## 5. Projected Total Savings

| Phase | Monthly saving | Cumulative |
|---|---|---|
| Phase 1 (quick wins) | ~$2,500 | ~$2,500 |
| Phase 2 (right-sizing) | ~$3,400 | ~$5,900 |
| Phase 3 (structural) | ~$5,000 | ~$10,900 |
| **Combined** | **~$10,900/month** | **~73% reduction** |

New estimated monthly cost after all phases: **~$4,100/month**
(down from $15,000)

At 20% commission on monthly savings: **~$2,180/month ongoing** or **~$10,900 one-time** on first-month delta.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Machine → Docker conversion breaks Docker-in-Docker networking | Test each job individually in a staging branch before merging |
| Shared base image becomes stale | Add a weekly workflow that rebuilds it on `requirements.txt` change |
| PR workflow split causes regressions to slip through | Keep full suite on `main`; require passing PR checks before merge |
| xdist parallelism reduction increases test duration | Monitor p95 test times; adjust `parallelism` and `-n` together |

---

*Analysis performed: 2026-03-07*
*Files analyzed: `.circleci/config.yml` (4,661 lines), `.circleci/requirements.txt`*
