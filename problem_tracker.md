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

| Batch       | Callbacks | avg           | p95    | >1s   |
| ----------- | --------- | ------------- | ------ | ----- |
| Worst cases | on        | ~0.45–0.54s   | ~1.2s  | 5–8%  |
| Control     | off       | ~0.30–0.38s   | ~0.4s  | ~1%   |

---

### Making the Issue Reproducible

#### Random vs Sequential User IDs

Random user IDs reliably triggered the issue:

| User IDs   | Callbacks | p95     | >1s    |
| ---------- | --------- | ------- | ------ |
| random     | on        | ~1.05s  | ~5%    |
| random     | off       | ~0.50s  | ~1.5%  |
| sequential | on/off    | ~0.39s  | ~0.8%  |

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

| Configuration   | >1s   |
| --------------- | ----- |
| Full Prometheus | 5.6%  |
| Early return    | 2.1%  |

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

| Metric   | On     | Off    | Δ     |
| -------- | ------ | ------ | ----- |
| Mean avg | 0.408s | 0.322s | +27%  |
| Mean p95 | 0.911s | 0.641s | +42%  |
| >1s      | 4.4%   | 1.7%   | 2.6×  |

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

| avg    | p95    | >1s   | >5s   |
| ------ | ------ | ----- | ----- |
| ~4.0s  | ~6.3s  | 100%  | ~33%  |

---

### Hypothesis

Baseline spikes are driven by:

* `end_user` DB lookups
* cache misses (especially for non-existent users)
* provider-side queuing under concurrency

---

### User Mode Experiments

#### 1000 requests / 100 users

| User mode  | avg    | p95    | >1s    |
| ---------- | ------ | ------ | ------ |
| none       | ~1.0s  | ~7.9s  | 10%    |
| sequential | ~1.5s  | ~8.6s  | ~32%   |
| random     | ~1.6s  | ~8.0s  | ~34%   |

**Key insight:**
Passing `user` dramatically increases latency.

---

### One Request vs Many Requests per User

#### Callbacks Off, Random Users

| Req/user | avg    | >1s   |
| -------- | ------ | ----- |
| 1        | ~8.6s  | 100%  |
| 10       | ~1.5s  | 33%   |

➡️ Cache hits on subsequent requests reduce latency by ~6×

---

### Created Users vs Random Users

| Mode    | avg (10/user) | >1s  |
| ------- | ------------- | ---- |
| random  | ~1.5s         | 33%  |
| created | ~1.2s         | 10%  |

---

### Full Test Comparison (callbacks off)

**1 request per user** (100 requests, 100 users):

| User mode | File                                  | avg     | p95     | max     | Above 1s |
| --------- | ------------------------------------- | ------- | ------- | ------- | -------- |
| none      | `user_none_1_requests_per_user.txt`    | 11.836s | 20.736s | 22.443s | 100%     |
| created   | `callbacks_off_1_per_user_created.txt` | 8.923s  | 14.962s | 15.774s | 100%     |
| random    | `callbacks_off_1_per_user.txt`         | 8.647s  | 14.909s | 15.668s | 100%     |

**10 requests per user** (1000 requests, 100 users):

| User mode             | File                                     | avg    | p95    | max     | Above 1s |
| --------------------- | ---------------------------------------- | ------ | ------ | ------- | -------- |
| none (callbacks off)  | `callbacks_off_user_none.txt`             | 0.998s | 7.966s | 14.698s | 10.0%    |
| none (callbacks on)   | `callbacks_on_user_none.txt`              | 1.099s | 7.767s | 14.873s | 10.0%    |
| created               | `callbacks_off_10_per_user_created.txt`   | 1.198s | 8.839s | 15.483s | 10.0%    |
| random                | `callbacks_off_10_per_user.txt`           | 1.538s | 8.275s | 15.499s | 33.2%    |

### Findings

The latency spike is high regardless of whether a user is passed to the payload or not.

---

### Use a local LLM mock provider

- [x] User none + caching true → `user_none_1_request_per_user_local_llm_cache_on.txt`
- [x] User none + caching false → `user_none_1_request_per_user_local_llm_cache_off.txt`

| Config                                             | File                                                | avg     | p95     | max     | Above 1s |
| -------------------------------------------------- | --------------------------------------------------- | ------- | ------- | ------- | -------- |
| User none, local LLM (localhost:8090), cache ON     | `user_none_1_request_per_user_local_llm_cache_on.txt`  | 15.204s | 26.543s | 27.706s | 100%     |
| User none, local LLM (localhost:8090), cache OFF    | `user_none_1_request_per_user_local_llm_cache_off.txt` | 15.449s | 26.317s | 27.430s | 100%     |

---

## What we know so far

1. **User parameter has no impact** - Passing a user to the request payload doesn't affect first request latency
2. **Cache has no impact** - Response caching ON vs OFF shows identical performance (~15s avg)
3. **Latency is proxy overhead** - 15s average suggests infrastructure (DB, Redis, auth) is the bottleneck, not LLM calls since the LLM server is running locally.

---

## Minimal Configuration Test Results

### Test with `test_config_minimal.yaml` (no DB, no Redis, no callbacks, no alerting)

- [x] **Test completed:** `validation_123.txt`

| Config                  | File                                                 | avg       | p95     | max     | Above 1s |
| ----------------------- | ---------------------------------------------------- | --------- | ------- | ------- | -------- |
| **Minimal config**      | `validation_123.txt`                                 | **8.654s**| 15.065s | 15.807s | 100%     |
| Full config (cache ON)  | `user_none_1_request_per_user_local_llm_cache_on.txt` | 15.204s   | 26.543s | 27.706s | 100%     |
| Full config (cache OFF) | `user_none_1_requests_per_user_local_llm_cache_off.txt` | 15.449s | 26.317s | 27.430s | 100%     |

**Improvement: 43% faster** (15.2s → 8.6s avg)

### Critical Findings

1. **Removing DB/Redis/callbacks cuts latency nearly in half**
   - Proves the 6.5s overhead comes from database operations, Redis, alerting infrastructure

2. **BUT: 8.6s is still very slow for a local mock LLM**
   - Local LLM mock likely responds in milliseconds
   - Proxy adds ~8.6s overhead **even with everything disabled**
   - Same wave pattern persists (early: 15s, mid: 6-8s, late: 1.5-4s)

3. **Root cause is in core proxy code, not features**
   - The bulk of latency comes from the proxy itself, not external network calls
   - Likely issues:
     - Request queueing/serialization (requests blocking each other)
     - Poor concurrency handling
     - Auth/router logic overhead
     - Python async event loop bottlenecks

---

## Conclusions

1. **User parameter:** No impact
2. **Cache:** No impact  
3. **Database/Redis/Alerting:** Adds ~6.5s overhead (15.2s → 8.6s)
4. **Core proxy code:** Adds ~8.6s overhead even with bare minimum config
5. **Total overhead:** ~15s for complex config processing 100 concurrent requests

**Next action:** Profile the proxy request path to identify the specific bottleneck in core code (likely concurrency/queueing)

## cProfile: Per-Request Hotspots (minimal config, no DB/Redis)

Profile captured with `test_config_minimal.yaml`. Ordered by cumtime, normalized per request:

| Component                            | Calls     | Cumtime | Per-call (est.) | Notes                               |
| ------------------------------------ | --------- | ------- | --------------- | ----------------------------------- |
| user_api_key_auth                    | 200       | 12.58s  | ~63ms           | Auth entry point                    |
| _user_api_key_auth_builder           | 100       | 12.55s  | **~125ms**      | Main auth logic                     |
| log_db_metrics wrapper               | 100       | 12.53s  | **~125ms**      | Still in path despite no DB         |
| FastAPI solve_dependencies           | 1600/200  | 12.61s  | ~63ms           | Dependency injection for chat route |
| Starlette/FastAPI middleware/routing | 600       | ~12.7s  | ~21ms           | Middleware stack                    |
| annotationlib call_annotate_function | 23400     | 8.28s   | ~0.4ms × many   | Python 3.14 annotation processing   |
| router acompletion                   | 586       | 0.67s   | ~1ms            | LLM call path (mostly I/O)          |

**Key finding:** Auth + log_db_metrics ≈ **125ms per request** even with no database. Expected for a simple master_key compare: microseconds.

**Optimization targets (in order):**
1. Auth path + log_db_metrics – ensure early return when master-key-only (no DB)
2. FastAPI dependency resolution – reduce number of injected dependencies on chat route
3. Lazy/conditional Prisma import – avoid loading Prisma when DB not configured

---

## Concurrency Test 1-10-30-50-100

**Setup:** Minimal config (`test_config_minimal.yaml`), no user in payload, master key auth, local LLM mock at localhost:8090. 100 total requests per run; concurrency = number of simulated users.

### Results

| Concurrency | File                               | avg    | p95     | max     | >1s  | >5s  | >10s |
| ----------- | ---------------------------------- | ------ | ------- | ------- | ---- | ---- | ---- |
| 1           | `concurrency_1_requests_100.txt`   | 0.012s | 0.007s  | 0.716s  | 0%   | 0%   | 0%   |
| 10          | `concurrency_10_requests_100.txt`  | 0.144s | 1.195s  | 2.034s  | 7%   | 0%   | 0%   |
| 30          | `concurrency_30_requests_100.txt`  | 0.833s | 3.861s  | 4.836s  | 27%  | 0%   | 0%   |
| 50          | `concurrency_50_requests_100.txt`  | 2.087s | 6.975s  | 7.689s  | 47%  | 18%  | 0%   |
| 100         | `concurrency_100_requests_100.txt` | 7.721s | 14.019s | 14.657s | 97%  | 69%  | 33%  |

**Δ vs baseline (c=1):**

| Concurrency | Δ avg              | Δ p95               | Δ max           |
| ----------- | ------------------ | ------------------ | --------------- |
| 1           | —                  | —                  | —               |
| 10          | +1,100% (12×)      | +16,971% (171×)    | +184% (2.8×)    |
| 30          | +6,842% (69×)      | +55,057% (551×)    | +575% (6.8×)    |
| 50          | +17,292% (174×)    | +99,529% (996×)    | +974% (10.7×)   |
| 100         | +64,242% (643×)    | +200,129% (2,002×) | +1,947% (20.5×) |

### Analysis

1. **Steep degradation with concurrency** — Average latency grows ~640× from 12 ms (c=1) to 7.7 s (c=100). p95 grows from 7 ms to 14 s.

2. **First request vs later requests** — At c=1, the first request is ~716 ms (cold/auth path), then ~5 ms per request. At c=10/30/50, the first request per user is 0.5–4 s, while subsequent requests are ~15–70 ms. At c=100, every request is effectively a “first” request (1 per user), so all pay the full contention cost.

3. **Evidence of queueing/serialization** — Higher concurrency yields much worse latency despite the same total load. This points to limited parallelism: requests appear to be processed in waves rather than truly in parallel.

4. **Tail latency** — At c=100, 33% of requests exceed 10 s. Worst-case (14.6 s) is far above the ~5 ms steady-state per-request cost, indicating substantial proxy overhead under load.

5. **Config context** — These runs use minimal config (no DB, Redis, callbacks) and the Prisma fast-path fix, so the bottleneck is in core proxy handling: auth, routing, and/or async event-loop behavior under concurrency.

### cProfile Collection

cProfile was run at each concurrency level with:
`$env:DATABASE_URL = $null; $env:REDIS_URL = $null; python -m cProfile -o concurrency_N_requests_100.prof -m litellm.proxy.proxy_cli --config test_config_minimal.yaml`
(Proxy runs until latency test completes 100 requests, then stopped.)

| Concurrency | Profile file                          |
| ----------- | ------------------------------------- |
| 1           | `concurrency_1_requests_100.prof`     |
| 10          | `concurrency_10_requests_100.prof`    |
| 30          | `concurrency_30_requests_100.prof`    |
| 50          | `concurrency_50_requests_100.prof`    |
| 100         | `concurrency_100_requests_100.prof`   |

- [x] `concurrency_1_requests_100.prof`
- [x] `concurrency_10_requests_100.prof`
- [x] `concurrency_30_requests_100.prof`
- [x] `concurrency_50_requests_100.prof`
- [x] `concurrency_100_requests_100.prof`

### cProfile Analysis

**Hypothesis (c=1 vs c=100):**

- **2× more completion calls per request** *(all OS)* – `router.acompletion` chain: ~309 calls (~3/req) at c=1 vs ~599 (~6/req) at c=100; suggests retries, fallbacks, or middleware work scaling with concurrency.
- **Thread pool teardown** *(all OS)* – c=100: ~14.7s in `threading.join` / ThreadPoolExecutor shutdown (29 threads); c=1: negligible. Suggests blocking work delegated to thread pool under load.
- **Context switching** *(all OS)* – `Context.run` ~15s cumtime at c=100 (7,503 calls); not in top 100 at c=1.

### Checklist

- [x] Concurrency 1 → `concurrency_1_requests_100.txt`
- [x] Concurrency 10 → `concurrency_10_requests_100.txt`
- [x] Concurrency 30 → `concurrency_30_requests_100.txt`
- [x] Concurrency 50 → `concurrency_50_requests_100.txt`
- [x] Concurrency 100 → `concurrency_100_requests_100.txt`

---

## Real Auth vs Mock Auth Comparison

**Setup:** Minimal config (`test_config_minimal.yaml`), no DB/Redis. Command: `poetry run python measure_latency.py --user-mode none --key-mode shared`. Instrumented with `[proxy] user_api_key_auth called at T` and `[proxy] chat/completions request received at T` prints.

### Log Files

| Auth mode | File                                  | Description                                                                 |
| --------- | ------------------------------------- | --------------------------------------------------------------------------- |
| Real      | `proxy_real_auth_100_requests_log.txt` | Full auth path (`_read_request_body`, `_user_api_key_auth_builder`)          |
| Mock      | `proxy_mock_auth_100_requests_log.txt` | Early return with mock `UserAPIKeyAuth`; skips DB/cache and body parsing     |

### Findings

| Metric                            | Real auth                             | Mock auth                |
| --------------------------------- | ------------------------------------- | ------------------------ |
| Auth calls before first chat      | 93                                    | 1 (interleaved)          |
| First auth → first chat handler   | ~47 ms                                | ~0.3–0.6 ms              |
| Auth↔chat pattern                 | Auth burst; chat handler delayed      | Auth and chat back-to-back |
| 100 requests throughput           | ~250 ms (auth dominates)              | ~152 ms                  |

### Interpretation

1. **Auth path is the bottleneck** — Real auth does 93 calls before the first request reaches the chat handler; mock auth processes each request immediately. The ~47 ms gap is consistent with cProfile (~63 ms per auth call under load).
2. **Queueing in real auth** — Auth invocations are serialized and block the chat route; mock auth removes that delay, so chat handlers run as soon as requests arrive.
3. **Confirms cProfile targets** — Aligns with cProfile: `user_api_key_auth` + `_user_api_key_auth_builder` + `log_db_metrics` ≈ 125 ms/req. Mocking auth eliminates that cost.