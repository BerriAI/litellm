# LiteLLM DB reset (claude-usage-proxy)

This repo builds and deploys the platform LiteLLM image (`deploy/values.yaml` → Argo app `claude-usage-proxy`). The live Postgres data and admin sync scripts live with the config repo.

**Canonical runbook:** [bitovi/claude-usage-proxy `DB_RESET.md`](https://github.com/bitovi/claude-usage-proxy/blob/main/DB_RESET.md)

Use that doc when master-key / salt encryption is unrecoverable and you need to wipe the `litellm` database on CNPG cluster `claude-usage-proxy-db`.

### Quick pointers (this repo)

| Concern | Where |
|---|---|
| Image tag, env (`LITELLM_MODE`, Redis host), OnePasswordItem, HPA, ReplicaSet history | `deploy/values.yaml` (branch `platform-deploy`) — `revisionHistoryLimit: 5` |
| ECR image retention | platform `ecr.lifecycle.maxImageCount` (default 50) |
| Model routing / `master_key: os.environ/…` | `bitovi/claude-usage-proxy` → `deploy/manifests/litellm_config.yaml` |
| `make sync-hub` after wipe | `bitovi/claude-usage-proxy` |

Salt key is **not** in config: set `LITELLM_SALT_KEY` on the 1Password item `claude-usage-proxy-secrets` only. Deployment `envFrom` injects it.
