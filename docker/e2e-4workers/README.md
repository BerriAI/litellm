# e2e-4workers

Isolated local Docker deployment of the litellm gateway at `num_workers=4`, for running `tests/e2e/` against it and comparing behavior against the `berrie-litellm-stage` EKS cluster. Stage runs the same `litellm_internal_staging` code but at 1 worker / 1 replica, so this stack isolates the multi-worker / shared-state surface.

## What this is not

Not the same topology as EKS. Three variables differ at once: 4 gunicorn UvicornWorkers in one container vs N single-worker pods, standalone Redis vs ElastiCache cluster+TLS, and local Postgres vs Aurora IAM. The useful signal is narrow: a test that passes on EKS-1-worker but fails here is a candidate multi-worker / shared-state bug (per-worker router reload, 4x budget rescheduler, `/metrics` multiproc aggregation, cross-worker cache coherence, the LIT-4909 fresh-worker-serves-400 path).

The image is built locally on arm64 (Apple Silicon); EKS runs amd64. Python behavior is arch-independent, so this is a behavior comparison, not a perf one.

## Build the gateway image from the staging ref

Both sides must run identical code, so build from `litellm_internal_staging`, not your working branch:

```bash
git fetch berri litellm_internal_staging
git worktree add --detach ../litellm-staging-wt berri/litellm_internal_staging
docker build -t litellm-gateway:staging-27d2fa84 ../litellm-staging-wt
```

The tag encodes the staging SHA; bump it when you rebuild from a newer staging commit and update `docker-compose.yml`.

## Run

Provider keys are read from `tests/e2e/.env` (the same file the suite loads host-side).

```bash
docker compose -f docker/e2e-4workers/docker-compose.yml up -d
docker compose -f docker/e2e-4workers/docker-compose.yml logs -f gateway   # look for "with 4 workers"
curl -fs http://localhost:4000/health/liveliness

LITELLM_PROXY_URL=http://localhost:4000 REDIS_CLUSTER=false REDIS_SSL=false \
  uv run pytest tests/e2e/llm_translation tests/e2e/quota_management -v
```

Tear down (drop the DB too):

```bash
docker compose -f docker/e2e-4workers/docker-compose.yml down -v
```
