# Deployment, Rollback, and Backup

Operational guide for the Bitovi LiteLLM fork (`bitovi/litellm`) and the proxy it feeds (`bitovi/claude-usage-proxy`).

## Before the first publish (required)

### 1. ECR repository must exist

The target repo is `claude-usage-proxy-litellm` in `us-west-2`. Create it by merging `terraform/ecr.tf` in `claude-usage-proxy` and applying Terraform, or create it manually in the AWS console.

### 2. GitHub secrets on `bitovi/litellm`

The publish workflow uses **IAM access keys** (same pattern as other Bitovi deploy workflows), not OIDC:

| Secret | Required | Purpose |
|--------|----------|---------|
| `AWS_ACCESS_KEY_ID` | Yes | IAM user access key with ECR push permissions |
| `AWS_SECRET_ACCESS_KEY` | Yes | Matching secret key |
| `PROXY_REPO_DISPATCH_TOKEN` | No | PAT to trigger ECS redeploy in claude-usage-proxy after publish |
| `LITELLM_ECR_REPOSITORY` (var) | No | Defaults to `claude-usage-proxy-litellm` |

`AWS_REGION` is set in the workflow to `us-west-2` (proxy infra region).

The IAM user needs at minimum: `ecr:GetAuthorizationToken` on `*` and push permissions on `arn:aws:ecr:us-west-2:<account-id>:repository/claude-usage-proxy-litellm`.

### 3. Publish

Run **Publish Bitovi Proxy Image** (or push to `litellm_internal_staging`). ECS cutover to ECR is a **later** step in claude-usage-proxy.

## Overview

Two repos share one AWS account:

| Repo | Responsibility |
|------|----------------|
| `bitovi/litellm` (this repo) | Build and push the forked LiteLLM Docker image to ECR |
| `bitovi/claude-usage-proxy` | Terraform (VPC, ECS, RDS, ECR, IAM), `litellm_config.yaml`, config deploy, ECS redeploy |

```
bitovi/litellm  --publish-->  ECR (claude-usage-proxy-litellm)
                                    |
                                    v pull
                              ECS Fargate (litellm service)
                                    |
                    litellm_config.yaml (S3) + Secrets Manager
```

**Region:** infrastructure runs in `us-west-2`. Terraform state lives in `us-east-1` (`s3://bitovi-claude-proxy-tfstate`).

## This repo's scope

This fork **only publishes** the Docker image via `.github/workflows/publish-bitovi-proxy-image.yml`.

Upstream LiteLLM workflows are preserved under `.github/workflows-backup/` and are **not** executed.

---

## AWS CLI access

SSO login alone is not enough; every command must use the profile you logged into.

```bash
aws sso login --profile Bitovi-ai
export AWS_PROFILE=Bitovi-ai

# If you still get InvalidClientTokenId, clear stale env vars:
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AWS_PROFILE=Bitovi-ai

aws sts get-caller-identity
```

Terraform outputs (run from `claude-usage-proxy`):

```bash
export AWS_PROFILE=Bitovi-ai
cd ../claude-usage-proxy/terraform   # or clone that repo separately
terraform init
terraform output github_actions_role_arn
terraform output ecr_litellm_repository_url
```

---

## GitHub Actions secrets

### `bitovi/litellm` (this repo)

| Secret / var | Required | Purpose |
|--------------|----------|---------|
| `AWS_ACCESS_KEY_ID` | Yes | IAM user with ECR push to `claude-usage-proxy-litellm` |
| `AWS_SECRET_ACCESS_KEY` | Yes | Matching secret key |
| `PROXY_REPO_DISPATCH_TOKEN` | No | PAT to trigger **Redeploy ECS** in `claude-usage-proxy` after publish |
| `LITELLM_ECR_REPOSITORY` (var) | No | Defaults to `claude-usage-proxy-litellm` |

Region is `us-west-2` (hardcoded in the workflow).

### `bitovi/claude-usage-proxy`

| Secret | Source |
|--------|--------|
| `AWS_ROLE_ARN` | `terraform output github_actions_role_arn` |
| `AWS_REGION` | `us-west-2` |
| `S3_BUCKET_NAME` | `terraform output config_bucket_name` |
| `ECS_CLUSTER_NAME` | `terraform output ecs_cluster_name` |
| `ECS_SERVICE_NAME` | `terraform output ecs_service_name` |

Workflows there: **Terraform** (`terraform/**` on `main`), **Deploy LiteLLM Config** (`litellm_config.yaml`), **Redeploy ECS** (manual or `repository_dispatch`).

---

## What gets deployed (three layers)

### 1. Infrastructure (Terraform) — `claude-usage-proxy`

Trigger: merge to `main` with changes under `terraform/`.

Creates/updates VPC, ALB, ECS, RDS, Redis, S3 config bucket, ECR, IAM (including GitHub OIDC), Secrets Manager placeholders.

### 2. Runtime config (no image change) — `claude-usage-proxy`

Trigger: merge to `main` with changes to `litellm_config.yaml`.

Uploads config to S3 and forces an ECS redeployment. Same Docker image; new models, budgets, MCP settings, etc.

### 3. LiteLLM image (fork binary) — this repo

Trigger: **Publish Bitovi Proxy Image** here.

Builds `docker/Dockerfile.database`, pushes to ECR as `:latest` and `:<git-sha>`. ECS only picks this up after redeploy in `claude-usage-proxy` (and only if the task definition points at ECR).

---

## Safe rollout: ECR cutover

Do not point ECS at ECR before a good image exists. Use phased merges in `claude-usage-proxy`.

### Phase A — additive infra (low risk)

Merge to `claude-usage-proxy` `main`:

- `terraform/ecr.tf` (creates ECR repo)
- `terraform/iam.tf` (OIDC trust for `bitovi/litellm`)
- `terraform/variables.tf` (`litellm_github_repo`)

**Keep production on the current image** until Phase C. Pin explicitly during apply:

```bash
cd claude-usage-proxy/terraform
terraform apply -var='litellm_image=ghcr.io/berriai/litellm:latest'
```

### Phase B — publish fork image (no ECS switch)

1. Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` on this repo
2. Run **Publish Bitovi Proxy Image** (or push to `litellm_internal_staging`)
3. Confirm tags in ECR:

```bash
export AWS_PROFILE=Bitovi-ai
aws ecr describe-images \
  --repository-name claude-usage-proxy-litellm \
  --region us-west-2 \
  --query 'imageDetails[*].imageTags' \
  --output table
```

### Phase C — cut over ECS (controlled)

In `claude-usage-proxy`, pin a **git SHA**, not `:latest`, for the first cutover:

```bash
cd claude-usage-proxy/terraform
terraform apply \
  -var='litellm_image=<account-id>.dkr.ecr.us-west-2.amazonaws.com/claude-usage-proxy-litellm:<git-sha>'
```

Verify:

```bash
curl -s https://llm-proxy.bitovi-ai.com/health/liveliness
```

### Pre-cutover checklist

Record the current task definition revision (your rollback target):

```bash
aws ecs describe-services \
  --cluster claude-usage-proxy \
  --services litellm \
  --region us-west-2 \
  --query 'services[0].taskDefinition' \
  --output text
```

---

## Backup strategy

| Asset | Backup mechanism | Retention / notes |
|-------|------------------|-------------------|
| **PostgreSQL (users, keys, spend)** | RDS automated backups | 7 days (`claude-usage-proxy/terraform/rds.tf`) |
| **RDS deletion** | `deletion_protection = true`; final snapshot on destroy | Snapshot id: `claude-usage-proxy-final` |
| **Terraform state** | S3 `bitovi-claude-proxy-tfstate`, encrypted | Enable S3 versioning on the bucket for point-in-time state recovery |
| **LiteLLM config** | Git (`litellm_config.yaml` in claude-usage-proxy) + S3 | Revert commit and merge to restore |
| **Docker images** | ECR tags per git SHA + `:latest` | Lifecycle keeps last **10** images |
| **ECS task definitions** | AWS retains every revision automatically | Roll back by re-pointing the service at an older revision |
| **Secrets** | Secrets Manager (`claude-usage-proxy/*`) | No automatic history; document values in a team vault when rotated |

**What an image swap does not destroy:** database rows, Redis state, S3 config, or secrets. A bad image causes task startup/health-check failures until you roll back the task definition or image pin.

**Recommended before risky fork upgrades**

1. Note current ECS task definition ARN/revision
2. Note current `litellm_image` value (ghcr URL or ECR SHA)
3. Ensure a recent RDS backup exists (default daily)
4. For schema risk, consider a manual RDS snapshot:

```bash
aws rds create-db-snapshot \
  --db-instance-identifier claude-usage-proxy \
  --db-snapshot-identifier claude-usage-proxy-pre-fork-$(date +%Y%m%d) \
  --region us-west-2
```

---

## Rollback procedures

### Rollback Docker image (fastest — minutes)

**Option 1: ECS task definition revision** (no Terraform)

```bash
export AWS_PROFILE=Bitovi-ai

aws ecs list-task-definitions \
  --family-prefix claude-usage-proxy \
  --sort DESC \
  --region us-west-2

aws ecs update-service \
  --cluster claude-usage-proxy \
  --service litellm \
  --task-definition claude-usage-proxy:42 \
  --force-new-deployment \
  --region us-west-2
```

**Option 2: Terraform pin to previous ECR SHA or ghcr** (`claude-usage-proxy`)

```bash
cd claude-usage-proxy/terraform
terraform apply -var='litellm_image=ghcr.io/berriai/litellm:latest'
# or
terraform apply -var='litellm_image=<ecr-url>:<previous-good-sha>'
```

**Option 3: GitHub Actions**

Run **Redeploy ECS** in `claude-usage-proxy` after Terraform already points at a good image tag.

### Rollback config only (`claude-usage-proxy`)

Revert `litellm_config.yaml` on `main` → **Deploy LiteLLM Config** runs automatically.

### Rollback Terraform infrastructure (`claude-usage-proxy`)

Revert the Terraform commit on `main` → **Terraform** workflow applies previous code.

### Rollback database (last resort)

Restore from RDS automated backup or manual snapshot. Image rollbacks do not require DB restore.

---

## Routine operations

### Publish new fork build (this repo)

1. Merge to `litellm_internal_staging`
2. Wait for **Publish Bitovi Proxy Image**
3. Redeploy ECS in `claude-usage-proxy` (auto via `PROXY_REPO_DISPATCH_TOKEN`, or **Redeploy ECS** / config push)

### Change models or budgets only (`claude-usage-proxy`)

Edit `litellm_config.yaml` → merge to `main`. No image rebuild required.

### Change infra only (`claude-usage-proxy`)

Edit `terraform/` → merge to `main` → review Terraform plan on the PR.

---

## Failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `InvalidClientTokenId` locally | Wrong/missing `AWS_PROFILE` or stale `AWS_*` env vars | `export AWS_PROFILE=Bitovi-ai` after `aws sso login` |
| `terraform output` empty locally | No `terraform init` against remote state | `terraform init` in `claude-usage-proxy/terraform` |
| ECS `CannotPullContainerError` | ECR empty or wrong tag | Publish image first; pin valid SHA |
| ECR push denied | Missing ECR repo or IAM user lacks push permissions | Apply `ecr.tf` or create repo; fix IAM policy |
| New image up but old behavior | ECS not redeployed | Force redeploy in claude-usage-proxy |

---

## Related docs

- [`.github/workflows-backup/README.md`](./.github/workflows-backup/README.md) — disabled upstream workflows
- [`bitovi/claude-usage-proxy` README](https://github.com/bitovi/claude-usage-proxy) — proxy operator setup, OAuth, virtual keys
