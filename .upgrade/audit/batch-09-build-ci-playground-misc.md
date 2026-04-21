# Batch 09 — Build / CI / Playground / misc

**Commits:** 12
**Scope:** Custom model-price/context entries, Jenkins file, GCR build pipeline, trunk-based-dev build, Prisma version bump, `litellm-proxy-extras` overlay, XYNE-123 playground booking integration.

## Upstream landscape (v1.81.3 → v1.83.3)

| File | # upstream commits | Notes |
|---|---|---|
| `model_prices_and_context_window.json` | **266** | Massive churn — new models added every week |
| `litellm/proxy/_types.py` | 145 | Very high churn |
| `litellm/proxy/proxy_server.py` | 183 | Very high |
| `pyproject.toml` | 102 | Frequent dependency bumps |
| `litellm/proxy/schema.prisma` | 68 | Schema migrations |
| `requirements.txt` | 62 | Deps |
| `litellm-proxy-extras/.../schema.prisma` | 57 | Schema |
| `schema.prisma` | 53 | Schema |
| `.circleci/config.yml` | 77 | CI config |
| `docker/Dockerfile.non_root` | 16 | Docker |
| `Dockerfile` | 13 | Docker |
| `.github/workflows/build-gcr.yml` | 0 | **Our file only** |
| `litellm/proxy/management_endpoints/playground_endpoints.py` | 0 | **Our file only** |

### Upstream equivalents

- **`litellm-proxy-extras` bumps**: multiple upstream commits bump the pinned version (`0.4.64`, `0.4.65`). Our XYNE-123 overlay (#130, #131) builds proxy-extras from source instead of PyPI — a different mechanism. No drop.
- **Prisma version bump**: no upstream commit matched "prisma version bump" search. Our bump is independent.
- **Custom model contexts**: upstream adds many models to `model_prices_and_context_window.json`; our entries are for juspay-specific models — no overlap in model names, but same file.

No DROPs in batch 09.

---

## Per-commit audit

### aa6a513627, c3d7591b89, f895a497e2 — custom model contexts (3 commits)

- **files:** `model_prices_and_context_window.json`
- **upstream:** 266 commits on this JSON.
- **decision:** **REWORK (merge-resolve)** — our 3 commits add custom model entries (juspay-branded aliases, public names). Upstream added many new models in the same file.
- **replay plan:** Cherry-pick all 3 in order; for each conflict, merge both JSON object sets (ours + upstream). Since JSON has stable keys, conflicts are usually mechanical (different keys added in different orders).
- **verification:** grep final JSON for juspay model keys, confirm present.

### 479cf2f557 — jenkins file

- **files:** `.github/workflows/build-gcr.yml`
- **upstream:** 0 commits on this file.
- **decision:** **KEEP-AS-IS**

### a85db92ca0 — sandbox branch for GCR push (#61)

- **files:** `.github/workflows/build-gcr.yml`
- **decision:** **KEEP-AS-IS**
- **rationale:** Stacked on #479cf2f557; same file.

### faf40db2db — build file changes for trunk-based dev (#85)

- **files:** `.github/workflows/build-gcr.yml`
- **decision:** **KEEP-AS-IS**

### 5a8826ee90 — bumping up prisma version (#113)

- **files:** `.circleci/config.yml` (77 upstream), `.github/workflows/build-gcr.yml` (0), `docker/Dockerfile.non_root` (16), `pyproject.toml` (102), `requirements.txt` (62)
- **decision:** **REWORK**
- **rationale:** Touches four churned config/build files. Upstream may have bumped Prisma separately or kept older version — must reconcile version pin.
- **replay plan:** Cherry-pick; for each file, determine whether our pin or upstream's is newer. Prefer upstream's unless our higher version is required for a runtime feature (verify from commit message).
- **verification:** `prisma --version` inside GCR image matches expected (MUST-SURVIVE #25).

### 7519a5cc19 — Playground integration (#124)

- **files:** migration `.sql` (new), 3 `schema.prisma` files (53/57/68 upstream commits), `_types.py` (145), `playground_endpoints.py` (new), `proxy_server.py` (183)
- **decision:** **REWORK (high effort)**
- **rationale:** Biggest commit in batch 09. New-file additions are clean (migration, endpoint module). But `schema.prisma` changes will conflict — upstream ran many migrations on the same schema. `_types.py` and `proxy_server.py` are very-high-churn.
- **replay plan:**
  1. Cherry-pick the new migration + `playground_endpoints.py` clean.
  2. Re-apply schema.prisma additions — the new playground-booking tables — after upstream's schema state. Ensure migration timestamps sequence correctly.
  3. Re-apply `_types.py` Pydantic model additions manually.
  4. Re-register the playground router in `proxy_server.py`.
- **verification:** MUST-SURVIVE #22 (playground booking workflow).

### 31e8feccd5 — overlay local litellm-proxy-extras over PyPI wheel (#130)

- **files:** `Dockerfile` (13), `litellm-proxy-extras/pyproject.toml` (likely low churn)
- **decision:** **REWORK (low effort)**
- **rationale:** Dockerfile has 13 upstream commits; our overlay layer is additive. Likely resolves easily.

### b4000796ea — build litellm-proxy-extras from source in GCR images (#131)

- **files:** `.github/workflows/build-gcr.yml` (0), `Dockerfile` (13), `docker/Dockerfile.non_root` (16)
- **decision:** **REWORK (low effort)**
- **rationale:** Same pattern as #130. Our GCR workflow file is untouched upstream.

### 7d639c60f7 — playground booking: seat-selection schema (#132)

- **files:** `_types.py` (145), `playground_endpoints.py` (our file)
- **decision:** **REWORK** (Pydantic model edits in churned file)
- **rationale:** Amends our playground models. Endpoint file changes are clean (our file).

### 22de914510 — let users retry failed/cancelled playground bookings (#133)

- **files:** `_types.py` (145), `playground_endpoints.py` (our file)
- **decision:** **REWORK**
- **rationale:** Same pattern as #132.

---

## Batch summary

| # | SHA | Decision | Risk |
|---|---|---|---|
| 1-3 | model contexts | REWORK (mechanical) | LOW-MED |
| 4-6 | build-gcr.yml additions | KEEP-AS-IS | LOW |
| 7 | 5a8826ee90 prisma bump | REWORK | MED |
| 8 | 7519a5cc19 playground | REWORK | HIGH |
| 9-10 | 31e8feccd5, b4000796ea | REWORK | LOW |
| 11-12 | 7d639c60f7, 22de914510 | REWORK | MED |

**No DROPs.** 4 KEEP-AS-IS, 8 REWORK. Main hotspot is the playground integration (#124) touching four churned files.

## Replay notes

- `model_prices_and_context_window.json` REWORK is mechanical — take both sets of keys; JSON conflict markers are easy to resolve since each model entry is a self-contained object.
- `schema.prisma` changes must be sequence-safe — ensure our migration's timestamp lands AFTER all upstream migrations carried in v1.83.3.
- Playground batch (#124, #132, #133) should be replayed together; don't split across sessions.
- After replay, run `prisma generate && prisma migrate status` to catch any migration ordering issues.
