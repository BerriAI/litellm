# Batch 03 — Routing + Vision

**Commits:** 17
**Scope:** Vision-model fallback (pre-request routing), least-busy routing Redis fix, sticky least-busy (both in-memory and Redis variants), usage-based-routing-v2 redis_only, simple-shuffle modifications, VLLM context-window error pattern, routing issue fixes, Prometheus metric for routing, debug-log cleanup.

**Criticality:** MUST-SURVIVE items #3, #4 (sticky least-busy), #5 (redis_only v2), #6 (simple-shuffle), #7 (vision fallback) depend on this batch. **Highest customization density in the whole upgrade.**

## Upstream landscape (v1.81.3 → v1.83.3)

| File | # upstream commits | Notes |
|---|---|---|
| `litellm/router.py` | **117** | Highest-churn file in our whole surface |
| `litellm/proxy/_types.py` | 145 | Very high |
| `litellm/proxy/litellm_pre_call_utils.py` | 51 | High |
| `litellm/litellm_core_utils/prompt_templates/common_utils.py` | 21 | Moderate |
| `litellm/types/router.py` | 14 | Low-mod |
| `litellm/types/integrations/prometheus.py` | 14 | Low-mod |
| `litellm/caching/dual_cache.py` | 6 | Low |
| `litellm/litellm_core_utils/exception_mapping_utils.py` | 4 | Low |
| `litellm/router_strategy/least_busy.py` | 1 | Minimal |
| `litellm/router_strategy/lowest_tpm_rpm_v2.py` | 2 | Minimal |
| `litellm/router_strategy/simple_shuffle.py` | 1 | Minimal |
| `litellm/router_strategy/sticky_least_busy.py` | 0 | **Our file** |
| `litellm/router_strategy/sticky_least_busy_redis.py` | 0 | **Our file** |
| `litellm/proxy/types_utils/utils.py` | 1 | Minimal |
| `litellm/proxy/management_endpoints/tag_management_endpoints.py` | 5 | Low |

### Upstream equivalents

- **Upstream sticky-sessions feature** (`e463fc22d9`, PR #21763): session-id-driven deployment affinity via `router_utils/pre_call_checks/deployment_affinity_check.py`. **Mechanism is different from our sticky-least-busy** — upstream routes by client-supplied session_id header; ours routes by first-call load balancing then pinning.
- **Upstream routing changes** on `router.py` (117 commits) and pre-call checks — extensive refactors to deployment selection, cooldown logic, and fallback chains. None of these replace our specific fixes.

### Drop-gate verdicts

| Custom feature | Upstream equivalent? | Safe to DROP? |
|---|---|---|
| Vision-model fallback routing | No — upstream has no non-vision-to-vision pre-request fallback | No |
| Sticky least-busy (in-mem) | No — upstream's sticky-sessions uses different mechanism | No |
| Sticky least-busy Redis variant | No | No |
| Least-busy Redis-only counter | No | No |
| usage-based-routing-v2 redis_only | No | No |
| Simple-shuffle modification | No | No |
| VLLM context-window error pattern | No | No |
| Routing fixes / Prometheus metric | No | No |

**No DROPs in batch 03.** Upstream's sticky-sessions is adjacent but not a replacement — we could in principle adopt upstream's `deployment_affinity_check` pattern as a substrate for our sticky-least-busy, but that's a REWORK, not a DROP.

---

## Per-commit audit (grouped by theme)

### Theme A — Vision-model fallback (5 commits)

| SHA | Subject | Files | Decision |
|---|---|---|---|
| a2534fd8c6 | python import based fallback | `types_utils/utils.py` (1) | **KEEP-AS-IS** |
| 46246fe236 | pre-request fallback for non-vision models | `_types.py` (145), `litellm_pre_call_utils.py` (51) | **REWORK** |
| 150ce42ad5 | testing fixes | `litellm_pre_call_utils.py` (51), `common_utils.py` (21) | **REWORK** |
| 03a998194b | vision support fix | `router.py` (117) | **REWORK** (highest-churn file) |
| 23447f1e0a | removed fuzzy check in vision model | `litellm_pre_call_utils.py` (51) | **REWORK** |

**Theme A verification:** MUST-SURVIVE item #7.

**Replay plan for Theme A:** resolve conflicts primarily in `litellm_pre_call_utils.py` — our fallback hook likely has shifted. `router.py` conflict (03a998194b) in the vision-detection branch requires care.

### Theme B — Least-busy / redis counter (4 commits)

| SHA | Subject | Files | Decision |
|---|---|---|---|
| 4a484f0378 | Fix/least busy routing (#16) | `least_busy.py` (1), test | **KEEP-AS-IS** |
| 15c5b875d2 | redis counter for load balancing (#19) | `dual_cache.py` (6), `least_busy.py` (1) | **KEEP-AS-IS or small REWORK** |
| b418c68ed1 | redis_only for usage-based-routing-v2 (#23) | `dual_cache.py` (6), `lowest_tpm_rpm_v2.py` (2) | **KEEP-AS-IS** |
| 338dcd82f2 | removed debug print logs (#123) | `auth_checks.py` (44), `least_busy.py` (1), `simple_shuffle.py` (1) | **REWORK** (auth_checks is high-churn) |

**Theme B verification:** MUST-SURVIVE items #3, #5, #9.

### Theme C — Sticky least-busy (4 commits)

| SHA | Subject | Files | Decision |
|---|---|---|---|
| e6e49c5069 | Feature/sticky least busy (#96) | `router.py` (117), `types/router.py` (14), `sticky_least_busy.py` (new), `build-gcr.yml` (0) | **REWORK** (router.py surface) |
| 2608c008fa | Feature/sticky busy redis (#105) | `router.py` (117), `types/router.py` (14), `sticky_least_busy.py`, `sticky_least_busy_redis.py` (new), `build-gcr.yml` (0) | **REWORK** (router.py surface) |
| b725ac3534 | Routing fixes + Prometheus metric (#110) | our sticky files (0), `types/integrations/prometheus.py` (14) | **REWORK** (prometheus types changed upstream) |
| ab851b9c2b | Fix/routing issues (#112) | our sticky files (0) | **KEEP-AS-IS** |

**Theme C verification:** MUST-SURVIVE items #3, #4.

**Replay plan for Theme C:** The `router.py` integration is the hard part. Find the current routing-strategy dispatch logic in v1.83.3's `router.py` and re-register `sticky-least-busy` and `sticky-least-busy-redis` routing strategies alongside upstream's. Consider whether to piggyback on upstream's `deployment_affinity_check` — note in audit, discuss with reviewer.

### Theme D — Simple shuffle + exception mapping (2 commits)

| SHA | Subject | Files | Decision |
|---|---|---|---|
| 22b9f44138 | modified simple shuffle routing (#74) | `simple_shuffle.py` (1) | **KEEP-AS-IS** |
| e72f67177b | VLLM context window error pattern (#28) | `exception_mapping_utils.py` (4), test | **KEEP-AS-IS** |

**Theme D verification:** MUST-SURVIVE item #6; `exception_mapping_utils` pattern covers VLLM-context-window → fallback trigger.

### Theme E — Miscellaneous routing fixes (2 commits)

| SHA | Subject | Files | Decision |
|---|---|---|---|
| 7d332c090f | fixes (#108) | `router.py` (117) | **REWORK** |
| 2973257d2d | Fix/oomkills (#117) | `tag_management_endpoints.py` (5) | **REWORK** |

---

## Batch summary

| # | Theme | Total | KEEP | REWORK | DROP |
|---|---|---|---|---|---|
| A | Vision fallback | 5 | 1 | 4 | 0 |
| B | Least-busy / redis counter | 4 | 3 | 1 | 0 |
| C | Sticky least-busy | 4 | 1 | 3 | 0 |
| D | Simple-shuffle + exception | 2 | 2 | 0 | 0 |
| E | Misc fixes | 2 | 0 | 2 | 0 |
| **Total** | | **17** | **7** | **10** | **0** |

**No DROPs.** 7 KEEP-AS-IS, 10 REWORK. `router.py` (117 upstream commits) is the dominant conflict surface — expect heaviest audit-intensity here.

## Replay notes

- **`router.py` is the hardest file in the whole upgrade.** Three commits in this batch modify it: `03a998194b`, `e6e49c5069`, `2608c008fa`, `7d332c090f`. Plan to resolve these with a dedicated session.
- For sticky-least-busy registration, study upstream's `deployment_affinity_check.py` first — it may be the cleanest pattern to mount our strategy on top of.
- **Do not split this batch across sessions.** Routing logic has cross-references between files; partial replays create confusing intermediate states.
- After replay, run:
  - `tests/local_testing/test_least_busy_routing.py`
  - `tests/test_litellm/router_strategy/test_sticky_least_busy*.py`
  - a 2-instance Redis soak test with sticky enabled (MUST-SURVIVE #4)
- If `router.py` conflicts exceed 300 lines during the first cherry-pick attempt, consider splitting batch 03 further (per-theme) and checkpointing between themes.
