### Context

**Issue:** Prometheus configured as a callback causes orders-of-magnitude worse performance (reported by Zurich).

**Reproduction:** Intermittent—requires many runs and configuration changes. Hypothesis: provider response speed may influence reproducibility.

### Next Steps

[x] {1} **Latency comparison (2 runs):** callbacks on vs off

**Run 1**
- [x] Callbacks on → `callbacks_on_3.txt`
- [x] Callbacks off → `callbacks_off_3.txt`

**Run 2**
- [x] Callbacks on → `callbacks_on_4.txt`
- [x] Callbacks off → `callbacks_off_4.txt`


#### Conclusion

**Data summary**

| Batch          | Callbacks | avg    | p95    | Above 1s |
|----------------|-----------|--------|--------|----------|
| 1 (yesterday)  | on        | 0.539s | 1.257s | 8.1%     |
| 1              | off       | 0.382s | 0.429s | 1.6%     |
| 2 (yesterday)  | on        | 0.453s | 0.472s | 3.4%     |
| 2              | off       | 0.383s | 0.450s | 1.0%     |
| 3 (today)      | on        | 0.410s | 0.499s | 0.8%     |
| 3              | off       | 0.338s | 0.425s | 0.8%     |
| 4 (today)      | on        | 0.374s | 0.438s | 0.8%     |
| 4              | off       | 0.316s | 0.391s | 0.8%     |

**Findings:** Yesterday (batches 1–2) showed a large impact: callbacks on had ~3× higher p95 and 5–8× more requests above 1s. Today (batches 3–4) shows only a modest overhead (~15–20% on avg/p95) with similar % above 1s. This supports the hypothesis that provider response speed influences reproducibility—faster responses today may have masked the issue.

[x] {2} Use local provider + artificial delays to test whether slower responses make the Prometheus callback issue more reproducible
**Run 1**
- [x] Callbacks on → `callbacks_on_5.txt`
- [x] Callbacks off → `callbacks_off_5.txt`

#### Conclusion

Adding artificial delays did not make the Prometheus callback impact more reproducible. The difference between callbacks on and off remained intermittent.

[x] {3} Let's increase the concurrency of the test and the total number of requests **(200 concurrent users / 15000)**
    **Run 1**
- [x] Callbacks on → `callbacks_on_6.txt`
- [x] Callbacks off → `callbacks_off_6.txt`

#### Conclusion

Higher concurrency and request number made performance worse for both cases.


[x] {4} Random ID generation for the user being passed onto the payload instead of sequential 1/2/3/4 ids.
    **Run 1** - Introduce the random ID generation
- [x] Callbacks on → `callbacks_on_7.txt`
- [x] Callbacks off → `callbacks_off_7.txt`
    **Run 2** - Remove the random ID generation
- [x] Callbacks on → `callbacks_on_8.txt`
- [x] Callbacks off → `callbacks_off_8.txt`
    **Run 3** - Bring back the random ID generation
- [x] Callbacks on → `callbacks_on_9.txt`
- [x] Callbacks off → `callbacks_off_9.txt`

#### Conclusion

**Data summary**

| Run   | User IDs   | Callbacks | avg    | p95    | Above 1s |
|-------|------------|-----------|--------|--------|----------|
| 1     | random     | on        | 0.415s | 1.014s | 5.1%     |
| 1     | random     | off       | 0.303s | 0.574s | 1.4%     |
| 2     | sequential | on        | 0.343s | 0.389s | 0.8%     |
| 2     | sequential | off       | 0.217s | 0.285s | 0.8%     |
| 3     | random     | on        | 0.484s | 1.052s | 5.4%     |
| 3     | random     | off       | 0.286s | 0.504s | 1.6%     |

**Findings:** With random user IDs, Prometheus callbacks clearly increase latency: ~5% of requests exceed 1s vs ~1.5% when callbacks are off. With sequential IDs, the difference disappears—both configs show ~0.8% above 1s. Random IDs make the Prometheus callback impact reproducible.

### Breakpoint

We now have a concrete reproduction. To maximize the chance of fixing Zurich's perf issues, we should address both:

1. **Prometheus overhead** — Extra latency when Prometheus callbacks are enabled.
2. **Baseline latency spikes** — Requests exceeding 1s even with callbacks off (~1.5% in our runs).



### Prometheus overhead

**Places to investigate (in order of suspicion):**

1. **`litellm/integrations/prometheus.py`**
   - `async_log_success_event` (line ~877) — main entry; runs on every completion
   - `_increment_remaining_budget_metrics` (line ~1171) — async; calls `_assemble_key_object`, `_assemble_team_object`, `_assemble_user_object`, which may hit DB/cache
   - `_assemble_user_object` (line ~2877) — fetches from DB when metadata is incomplete
   - `_assemble_team_object` (line ~2662) — calls `get_team_object` (DB/cache)
   - `_assemble_key_object` (line ~2807) — fetches key from cache/DB

2. **`prometheus_client` usage**
   - `.labels(**kwargs).inc()` / `.observe()` / `.set()` — sync, may contend on the registry lock when many label combinations are used
   - High cardinality from `end_user`, `user_api_key`, `model_id`, etc. can create many series

3. **Callback invocation**
   - `litellm/litellm_core_utils/litellm_logging.py` (line ~2569) — callbacks run sequentially; Prometheus blocks the response until it finishes

4. **`prometheus_system` callback** (if relevant)
   - `litellm/integrations/prometheus_services.py` — service-level metrics (Redis, Postgres); separate from completion flow

**Action Plan**

1. Early return from `async_log_success_event` to confirm Prometheus is the cause.

   | Config              | File                    | avg    | p95    | Above 1s |
   |---------------------|-------------------------|--------|--------|----------|
   | Callbacks on + early return | `callbacks_on_10a.txt`  | 0.361s | 0.631s | 2.1%     |
   | Callbacks on + full Prometheus | `callbacks_on_10b.txt` | 0.463s | 1.059s | 5.6%     |

   **Conclusion:** Early return reduces requests above 1s from 5.6% to 2.1%. Prometheus callback is confirmed as the source of the overhead.

2. **Bisect: async vs sync work** — Narrow down whether the overhead comes from async DB/cache lookups or sync metric updates.

   **Test A:** Early return *after* `_increment_remaining_budget_metrics` (skip sync metric updates). If latency improves → sync Prometheus updates are the culprit.

    **Run 1** - Introduce the random ID generation
    - [x] Callbacks on + early return → `callbacks_on_11a.txt`
    - [x] Callbacks on + no early return → `callbacks_on_11b.txt`

    **Conclusion:** No effect. Skipping sync metric updates did not improve latency—sync Prometheus updates are not the culprit.

   **Test B:** Early return *before* `_increment_remaining_budget_metrics` (skip async budget metrics only). If latency improves → DB/cache lookups in budget metrics are the culprit.

    **Run 1** - Introduce the random ID generation
    - [x] Callbacks on + early return → `callbacks_on_12a.txt`
    - [x] Callbacks on + no early return → `callbacks_on_12b.txt`

    **Conclusion:** Early return reduced requests above 1s from 4.8% to 1.9%. Root cause: `_increment_remaining_budget_metrics` (DB/cache lookups for key, team, and user budget).

3. **Isolate: comment out `_increment_remaining_budget_metrics`** — Run everything else (including the sync metric updates below). Confirms whether the function is the main cause or if the code below contributes.

   - [x] Comment out the function call only, run everything below → `callbacks_on_13a.txt`
   - [x] Bring the function back on => `callbacks_on_13b.txt`

    **Conclusion:** Commenting out that function stoped the latency issue caused by prometheus.

4. **Patch: run `_increment_remaining_budget_metrics` off critical path** — Use `asyncio.create_task()` instead of `await` so budget metrics run in background and no longer block request latency.

   **Run 1**
   - [x] Callbacks on + patch (create_task) → `callbacks_on_14a.txt`
   - [x] Callbacks on + no patch (baseline) → `callbacks_on_14b.txt`

   | Config   | File                    | avg    | p95    | Above 1s |
   |----------|-------------------------|--------|--------|----------|
   | Patch    | `callbacks_on_14a.txt`  | 0.468s | 0.911s | 4.4%     |
   | Baseline | `callbacks_on_14b.txt`  | 0.362s | 0.784s | 3.4%     |

   **Conclusion:** In this run, the patch did not improve latency—baseline (3.4% above 1s) outperformed the patch (4.4%). Run-to-run variance may be a factor; additional runs recommended to confirm.

5. **Baseline: 10 runs with Prometheus on, 10 with Prometheus off** — Measure latency to establish a strong baseline before testing the parallelization patch.

   **Prometheus on**
   - [x] Run 1 → `callbacks_on_15_1.txt`
   - [x] Run 2 → `callbacks_on_15_2.txt`
   - [x] Run 3 → `callbacks_on_15_3.txt`
   - [x] Run 4 → `callbacks_on_15_4.txt`
   - [x] Run 5 → `callbacks_on_15_5.txt`
   - [x] Run 6 → `callbacks_on_15_6.txt`

   | Run | File                    | avg    | p95    | A1s   | A2s   | A3s   | A4s   | A5s   | A6s   | A7s   | A8s   |
   |-----|-------------------------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|
   | 1   | `callbacks_on_15_1.txt` | 0.405s | 0.822s | 3.1%  | 1.1%  | 0.9%  | 0.7%  | 0.5%  | 0.3%  | 0.2%  | 0.0%  |
   | 2   | `callbacks_on_15_2.txt` | 0.442s | 0.887s | 4.2%  | 1.7%  | 1.0%  | 0.8%  | 0.6%  | 0.4%  | 0.3%  | 0.1%  |
   | 3   | `callbacks_on_15_3.txt` | 0.511s | 1.131s | 6.3%  | 1.9%  | 1.2%  | 0.9%  | 0.6%  | 0.4%  | 0.3%  | 0.1%  |
   | 4   | `callbacks_on_15_4.txt` | 0.381s | 0.952s | 4.6%  | 1.5%  | 0.9%  | 0.7%  | 0.5%  | 0.3%  | 0.2%  | 0.0%  |
   | 5   | `callbacks_on_15_5.txt` | 0.318s | 0.565s | 2.2%  | 1.1%  | 0.8%  | 0.6%  | 0.4%  | 0.3%  | 0.2%  | 0.0%  |
   | 6   | `callbacks_on_15_6.txt` | 0.393s | 1.106s | 5.7%  | 1.9%  | 1.0%  | 0.7%  | 0.4%  | 0.3%  | 0.2%  | 0.0%  |

   **Merged (6 runs, 30k requests):** mean avg 0.408s, mean p95 0.911s. 
    Above 1s: 1307 (4.4%), 2s: 455 (1.5%), 3s: 292 (1.0%), 4s: 213 (0.7%), 5s: 152 (0.5%), 6s: 101 (0.3%), 7s: 62 (0.2%), 8s: 18 (0.1%), 9s: 1 (0.0%), 10s: 0 (0.0%).

   **Prometheus off**
   - [x] Run 1 → `callbacks_off_15_1.txt`
   - [x] Run 2 → `callbacks_off_15_2.txt`
   - [x] Run 3 → `callbacks_off_15_3.txt`
   - [x] Run 4 → `callbacks_off_15_4.txt`
   - [x] Run 5 → `callbacks_off_15_5.txt`
   - [x] Run 6 → `callbacks_off_15_6.txt`

   | Run | File                     | avg    | p95    | A1s   | A2s   | A3s   | A4s   | A5s   | A6s   | A7s   | A8s   |
   |-----|--------------------------|--------|--------|-------|-------|-------|-------|-------|-------|-------|-------|
   | 1   | `callbacks_off_15_1.txt` | 0.319s | 0.693s | 2.0%  | 0.9%  | 0.8%  | 0.6%  | 0.5%  | 0.4%  | 0.3%  | 0.1%  |
   | 2   | `callbacks_off_15_2.txt` | 0.328s | 0.696s | 1.8%  | 0.9%  | 0.8%  | 0.6%  | 0.4%  | 0.3%  | 0.2%  | 0.0%  |
   | 3   | `callbacks_off_15_3.txt` | 0.276s | 0.540s | 1.2%  | 0.9%  | 0.7%  | 0.6%  | 0.4%  | 0.3%  | 0.2%  | 0.0%  |
   | 4   | `callbacks_off_15_4.txt` | 0.376s | 0.741s | 2.1%  | 0.9%  | 0.8%  | 0.6%  | 0.5%  | 0.4%  | 0.2%  | 0.1%  |
   | 5   | `callbacks_off_15_5.txt` | 0.347s | 0.580s | 1.7%  | 0.9%  | 0.8%  | 0.6%  | 0.5%  | 0.3%  | 0.2%  | 0.0%  |
   | 6   | `callbacks_off_15_6.txt` | 0.288s | 0.598s | 1.5%  | 0.9%  | 0.7%  | 0.6%  | 0.4%  | 0.3%  | 0.1%  | 0.0%  |

   **Merged (6 runs, 30k requests):** mean avg 0.322s, mean p95 0.641s. 
   Above 1s: 521 (1.7%), 2s: 271 (0.9%), 3s: 225 (0.8%), 4s: 180 (0.6%), 5s: 139 (0.5%), 6s: 98 (0.3%), 7s: 59 (0.2%), 8s: 15 (0.1%), 9s: 0 (0.0%), 10s: 0 (0.0%).

   **Comparison (Prometheus on vs off)**

   | Metric       | Prometheus on | Prometheus off | Δ        |
   |--------------|---------------|----------------|----------|
   | Mean avg     | 0.408s        | 0.322s         | +27%     |
   | Mean p95     | 0.911s        | 0.641s         | +42%     |
   | Above 1s     | 1307 (4.4%)   | 521 (1.7%)     | +2.6×    |
   | Above 2s     | 455 (1.5%)    | 271 (0.9%)     | +1.7×    |
   | Above 3s     | 292 (1.0%)    | 225 (0.8%)     | +1.3×    |

   Requests above threshold (bar length ∝ %):

   ```
   Above 1s   on  ██████████████████████ 4.4%   off  ████████ 1.7%
   Above 2s   on  ███████ 1.5%                  off  ████ 0.9%
   Above 3s   on  █████ 1.0%                    off  ████ 0.8%
   ```

   **Expectation (targets when root cause is fixed)**

   After addressing `_increment_remaining_budget_metrics` (e.g. parallelization, caching, or moving off critical path), Prometheus-on should approach Prometheus-off performance:

   | Metric     | Current (on) | Target (on)  | Baseline (off) |
   |------------|--------------|--------------|----------------|
   | Mean avg   | 0.408s       | ≤ 0.35s      | 0.322s         |
   | Mean p95   | 0.911s       | ≤ 0.70s      | 0.641s         |
   | Above 1s   | 4.4%         | ≤ 2.5%       | 1.7%           |
   | Above 2s   | 1.5%         | ≤ 1.0%       | 0.9%           |
   | Above 3s   | 1.0%         | ≤ 0.9%       | 0.8%           |

   Success = Prometheus-on metrics within ~10–15% of Prometheus-off, so enabling Prometheus has minimal impact on user-facing latency.

   **Conclusion:**

6. [x] **Patch: parallelize the three budget metric lookups** — Use `asyncio.gather()` instead of sequential `await` so key, team, and user lookups run in parallel.


### Final Conclusion

The issue was caused by hitting the database on every request when Prometheus was used as a callback. Commit `30534d7e82` fixes this by setting `check_db_only` to `False`.

> It was hard to diagnose because Prometheus was catching database errors and only logging them at debug level. Fixed by commit `d37796662` (adds `_log_budget_lookup_failure` to surface errors). This error is very important—it stops the cache from working properly.

## Baseline Latency Spikes (Separate from Prometheus)

**Issue:** Zurich reports huge latency spikes even when callbacks are off. Unlike the Prometheus callback issue (which adds ~2.6× latency when enabled), this baseline behavior affects both configs roughly equally.

**Reference data:** 42 requests, 42 users, fire-as-fast-as-possible

| Config          | avg    | p95    | Above 1s | Above 5s |
|-----------------|--------|--------|----------|----------|
| Callbacks off   | 4.034s | 6.345s | 100%     | 33.3%    |
| Callbacks on    | 4.183s | 6.620s | 100%     | 35.7%    |


**Reference files:** `callbacks_off_baseline.txt`, `callbacks_on_baseline.txt`

### Next Steps

**Context:** Callbacks on vs off no longer affects latency (Prometheus fix). The remaining baseline spikes are likely from end_user lookups, provider latency, or other bottlenecks. Comparing user modes will isolate whether passing `user` (and cache behavior) contributes.

**Measure baseline latency with each user mode** (run `measure_latency.py`):

| Mode        | Command                     | What it isolates                                                         |
|-------------|-----------------------------|--------------------------------------------------------------------------|
| `none`      | `--user-mode none`          | No end_user lookup; pure proxy + provider latency                         |
| `sequential`| `--user-mode sequential`    | End_user lookup with IDs 1,2,3,… (reused per user → cache hits)          |
| `random`    | `--user-mode random`        | End_user lookup with unique random IDs per user → more cache misses      |

**Callbacks on** (Prometheus enabled in proxy config):

- [x] `--user-mode none` → `callbacks_on_user_none.txt`
- [x] `--user-mode sequential` → `callbacks_on_user_sequential.txt`
- [x] `--user-mode random` → `callbacks_on_user_random.txt`

**Callbacks off** (Prometheus disabled in proxy config):

- [x] `--user-mode none` → `callbacks_off_user_none.txt`
- [x] `--user-mode sequential` → `callbacks_off_user_sequential.txt`
- [x] `--user-mode random` → `callbacks_off_user_random.txt`

**Compare:** If `none` shows much lower latency than `random`/`sequential`, end_user DB lookups are a contributor. If all three are similar, the bottleneck is elsewhere.

#### Analysis (1000 requests, 100 users)

**Data summary**

| Config        | User mode   | avg    | p95    | Above 1s |
|---------------|-------------|--------|--------|----------|
| Callbacks on  | none        | 1.099s | 7.767s | 10.0%    |
| Callbacks on  | sequential  | 1.702s | 8.524s | 37.7%    |
| Callbacks on  | random      | 1.773s | 8.598s | 42.5%    |
| Callbacks off | none        | 0.998s | 7.966s | 10.0%    |
| Callbacks off | sequential  | 1.532s | 8.626s | 31.8%    |
| Callbacks off | random      | 1.556s | 8.036s | 34.3%    |

**Findings**

1. **Passing `user` adds significant latency.** `none` has ~10% of requests above 1s vs ~32–43% when `user` is passed. Avg latency with `none` (~1.0–1.1s) is ~35–55% lower than with `sequential`/`random` (~1.5–1.8s).
2. **End_user lookups are a contributor.** The large gap between `none` and the other modes points to end_user DB/cache lookups adding measurable overhead.
3. **Random vs sequential:** Random is slightly worse (avg +0.2s, Above 1s +3–8%), consistent with more cache misses for unique IDs.
4. **Callbacks on vs off:** Small difference (~5–10%). Callbacks-on is marginally slower; callbacks-off is not the main driver of the baseline spikes.