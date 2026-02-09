# Zurich Performance Investigation

This document tracks two **independent but compounding issues** discovered during performance testing:

1. **Prometheus callback overhead** — latency added only when Prometheus is enabled
2. **Baseline latency spikes** — large spikes that occur even when callbacks are disabled

---

## Problem 1: Prometheus Callback Overhead

### Summary

**Issue:**
When Prometheus is configured as a callback, request latency increases significantly. In reproducible scenarios, Prometheus increases:

* mean latency by ~27%
* p95 latency by ~42%
* requests >1s by ~2.6×

This issue was intermittent at first but became fully reproducible with random user IDs.

---

### Reproduction Strategy

#### Baseline: Callbacks On vs Off

Multiple test batches comparing Prometheus enabled vs disabled:

**Key finding**

* Early batches showed severe impact
* Later batches showed smaller overhead
* Hypothesis formed: provider speed + cache behavior affects reproducibility

| Batch       | Callbacks | avg         | p95   | >1s  |
| ----------- | --------- | ----------- | ----- | ---- |
| Worst cases | on        | ~0.45–0.54s | ~1.2s | 5–8% |
| Control     | off       | ~0.30–0.38s | ~0.4s | ~1%  |

---

### Making the Issue Reproducible

#### Random vs Sequential User IDs

Random user IDs reliably triggered the issue:

| User IDs   | Callbacks | p95    | >1s   |
| ---------- | --------- | ------ | ----- |
| random     | on        | ~1.05s | ~5%   |
| random     | off       | ~0.50s | ~1.5% |
| sequential | on/off    | ~0.39s | ~0.8% |

**Conclusion:**
Prometheus overhead becomes reproducible when user lookups cause cache misses.

---

### Root Cause Analysis

#### Suspected Areas

1. **`_increment_remaining_budget_metrics`**

   * Performs DB/cache lookups for key, team, and user
   * Executed on every request
2. **Callback execution model**

   * Callbacks run sequentially and block response completion
3. **Prometheus label cardinality**

   * High-cardinality labels amplify contention

---

### Isolation & Bisection

#### Early Return Tests

| Configuration   | >1s  |
| --------------- | ---- |
| Full Prometheus | 5.6% |
| Early return    | 2.1% |

➡️ Confirms Prometheus callback is on the critical path.

---

#### Async vs Sync Breakdown

* Skipping **sync metric updates** → **no improvement**
* Skipping **budget metrics DB lookups** → **major improvement**

➡️ Root cause isolated to `_increment_remaining_budget_metrics`

---

### Fix Validation

#### Confirming the Cause

* Commenting out `_increment_remaining_budget_metrics` **eliminated the latency spike**
* Moving it off the critical path with `create_task()` showed mixed results due to variance

---

### Baseline Comparison (6 runs, 30k requests)

#### Prometheus ON vs OFF

| Metric   | On     | Off    | Δ    |
| -------- | ------ | ------ | ---- |
| Mean avg | 0.408s | 0.322s | +27% |
| Mean p95 | 0.911s | 0.641s | +42% |
| >1s      | 4.4%   | 1.7%   | 2.6× |

**Target after fix:** Prometheus-on within ~10–15% of Prometheus-off.

---

### Final Resolution

* Root cause: **DB lookups on every request inside Prometheus budget metrics**
* Cache was silently failing due to suppressed DB errors
* Fixes:

  * Disable `check_db_only`
  * Surface budget lookup failures in logs
  * Parallelize budget lookups

**Fix commits**

* `30534d7e82` — cache behavior fix
* `d37796662` — error visibility

---

## Problem 2: Baseline Latency Spikes (Independent of Prometheus)

### Summary

**Issue:**
Even with Prometheus fully disabled, Zurich sees extreme latency spikes (p95 >6s, 100% >1s).

This behavior:

* Affects callbacks on and off equally
* Is unrelated to the Prometheus fix
* Scales with end_user lookup patterns

---

### Reference Baseline (Callbacks Off)

| avg   | p95   | >1s  | >5s  |
| ----- | ----- | ---- | ---- |
| ~4.0s | ~6.3s | 100% | ~33% |

---

### Hypothesis

Baseline spikes are driven by:

* `end_user` DB lookups
* cache misses (especially for non-existent users)
* provider-side queuing under concurrency

---

### User Mode Experiments

#### 1000 requests / 100 users

| User mode  | avg   | p95   | >1s  |
| ---------- | ----- | ----- | ---- |
| none       | ~1.0s | ~7.9s | 10%  |
| sequential | ~1.5s | ~8.6s | ~32% |
| random     | ~1.6s | ~8.0s | ~34% |

**Key insight:**
Passing `user` dramatically increases latency.

---

### One Request vs Many Requests per User

#### Callbacks Off, Random Users

| Req/user | avg   | >1s  |
| -------- | ----- | ---- |
| 1        | ~8.6s | 100% |
| 10       | ~1.5s | 33%  |

➡️ Cache hits on subsequent requests reduce latency by ~6×

---

### Created Users vs Random Users

| Mode    | avg (10/user) | >1s |
| ------- | ------------- | --- |
| random  | ~1.5s         | 33% |
| created | ~1.2s         | 10% |

---

### Full Test Comparison (callbacks off)

**1 request per user** (100 requests, 100 users):

| User mode | File | avg | p95 | max | Above 1s |
|-----------|------|-----|-----|-----|----------|
| none | `user_none_1_requests_per_user.txt` | 11.836s | 20.736s | 22.443s | 100% |
| created | `callbacks_off_1_per_user_created.txt` | 8.923s | 14.962s | 15.774s | 100% |
| random | `callbacks_off_1_per_user.txt` | 8.647s | 14.909s | 15.668s | 100% |

**10 requests per user** (1000 requests, 100 users):

| User mode | File | avg | p95 | max | Above 1s |
|-----------|------|-----|-----|-----|----------|
| none (callbacks off) | `callbacks_off_user_none.txt` | 0.998s | 7.966s | 14.698s | 10.0% |
| none (callbacks on) | `callbacks_on_user_none.txt` | 1.099s | 7.767s | 14.873s | 10.0% |
| created | `callbacks_off_10_per_user_created.txt` | 1.198s | 8.839s | 15.483s | 10.0% |
| random | `callbacks_off_10_per_user.txt` | 1.538s | 8.275s | 15.499s | 33.2% |

### Findings

The latency spike is high regardless of whether a user is passed to the payload or not.

---

### Use a local LLM mock provider

- [x] User none + caching true → `user_none_1_request_per_user_local_llm_cache_on.txt`
- [x] User none + caching false → `user_none_1_request_per_user_local_llm_cache_off.txt`

| Config | File | avg | p95 | max | Above 1s |
|--------|------|-----|-----|-----|----------|
| User none, local LLM (localhost:8090), cache ON | `user_none_1_request_per_user_local_llm_cache_on.txt` | 15.204s | 26.543s | 27.706s | 100% |
| User none, local LLM (localhost:8090), cache OFF | `user_none_1_request_per_user_local_llm_cache_off.txt` | 15.449s | 26.317s | 27.430s | 100% |

---

## What we know so far

1. **User parameter has no impact** - Passing a user to the request payload doesn't affect first request latency
2. **Cache has no impact** - Response caching ON vs OFF shows identical performance (~15s avg)
3. **Latency is proxy overhead** - 15s average suggests infrastructure (DB, Redis, auth) is the bottleneck, not LLM calls since the LLM server is running locally.
