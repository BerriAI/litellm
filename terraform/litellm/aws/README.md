# LiteLLM on AWS (ECS Fargate)

Deploys the componentized LiteLLM proxy on AWS:

- **VPC** with public + private subnets across the AZs you pass in, one NAT gateway
- **Aurora Postgres** cluster — one writer instance + one reader instance, **IAM database authentication enabled**
- **ElastiCache Redis** (private, replication group with multi-AZ failover and at-rest + in-transit encryption) for caching + rate limiting
- **S3 bucket** (private, versioned, SSE-S3) — exposed to gateway + backend as `S3_BUCKET_NAME` / `S3_REGION_NAME` for cache backend, request log archival, and `/v1/files` storage
- **Secrets Manager** entries for `LITELLM_MASTER_KEY` (auto-generated, `sk-…`) and the Aurora master password (bootstrap-only)
- **ECS Fargate cluster** running three services — `gateway`, `backend`, `ui`
- **Application Load Balancer** (public, HTTP/80) with path-based routing:
  - LLM data-plane prefixes (`/v1/chat/*`, `/v1/embeddings`, …) → `gateway`
  - UI assets (`/`, `/_next/*`, `/litellm-asset-prefix/*`, …) → `ui`
  - Everything else (management API: `/key/*`, `/user/*`, …) → `backend`
- **One-off migration task** (`litellm-migrations`) that runs `prisma migrate deploy` from the dedicated `ghcr.io/berriai/litellm-migrations` image

## Aurora + IAM auth

The cluster runs with `iam_database_authentication_enabled = true`. Enabling
that on the cluster doesn't by itself let any Postgres user log in with an IAM
token — you also need to `CREATE USER ... GRANT rds_iam` once. `bootstrap.tf`
does this automatically during `terraform apply` via a one-shot Fargate task
(`postgres:16-alpine` running the bootstrap SQL with the master password from
Secrets Manager). The SQL is idempotent, so re-applies are safe.

The same apply also runs the prisma schema migration via the existing
`litellm-migrations` task definition, and the gateway/backend services
`depends_on` the migration so they don't start until the schema is in place.

At runtime, the proxy assembles `DATABASE_URL` from `DATABASE_HOST/PORT/USER/NAME`
plus a short-lived IAM token — see `litellm/proxy/auth/rds_iam_token.py`. The
task role has `rds-db:connect` scoped to the IAM-authed user on the cluster.

**Break-glass.** If you need to run the bootstrap or migration by hand (e.g.,
to re-apply against an externally provisioned cluster), `db_bootstrap_sql` and
`migration_run_command` are still exposed as outputs.

**Prerequisite.** `terraform apply` shells out to `aws ecs run-task` /
`aws ecs wait` in `local-exec` provisioners, so the machine running terraform
needs the `aws` CLI installed and authenticated.

## Configuring the proxy

### `proxy_config` (preferred)

Mirrors the helm chart's `gateway.config.proxy_config`. The map is YAML-encoded
and base64-passed to gateway, backend, and the migration task; each container
decodes it to `/tmp/litellm-config.yaml` at startup and sets `CONFIG_FILE_PATH`
to match.

```hcl
proxy_config = {
  model_list = [
    {
      model_name = "gpt-4o"
      litellm_params = {
        model   = "openai/gpt-4o"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
  ]
  general_settings = {
    master_key   = "os.environ/LITELLM_MASTER_KEY"
    database_url = "os.environ/DATABASE_URL"
  }
}
```

LiteLLM resolves `os.environ/<NAME>` references in the YAML against the
container's environment. That means provider API keys belong in
`*_extra_secrets` (next section), and your YAML just references them by name.

### Extra env vars

Non-sensitive plaintext (feature flags, observability hosts, etc.):

```hcl
gateway_extra_env = {
  LANGFUSE_HOST = "https://us.cloud.langfuse.com"
}
backend_extra_env = {
  STORE_MODEL_IN_DB = "True"
}
```

### Extra secrets (API keys)

Sensitive values — provider API keys, third-party tokens — live in **existing
Secrets Manager secrets**. Reference them by ARN:

```hcl
gateway_extra_secrets = {
  OPENAI_API_KEY    = "arn:aws:secretsmanager:us-west-2:111122223333:secret:openai-api-key-AbCdEf"
  ANTHROPIC_API_KEY = "arn:aws:secretsmanager:us-west-2:111122223333:secret:anthropic-api-key-GhIjKl"
}
```

What happens under the hood:
- The execution role auto-gains `secretsmanager:GetSecretValue` on every ARN
  listed here.
- ECS resolves each secret at task launch and injects its value into the
  container as the env var named on the left.
- The `proxy_config` YAML references the resulting env var via
  `os.environ/OPENAI_API_KEY`.

To pluck a single field out of a JSON secret, use ECS's `:fieldName::` suffix:

```hcl
gateway_extra_secrets = {
  OPENAI_API_KEY = "arn:…:secret:provider-keys-AbCdEf:openai_api_key::"
}
```

To create the secret beforehand:

```bash
aws secretsmanager create-secret \
  --name openai-api-key \
  --secret-string "sk-proj-..."
```

## Tenant deployment

Every resource the stack creates is named `${tenant}-litellm-${env}` (or
that plus a per-resource suffix), so multiple tenants and multiple
environments coexist in the same account as long as the `(tenant, env)`
pair differs:

| `tenant` | `env`   | Example resource name              |
| -------- | ------- | ---------------------------------- |
| `acme`   | `stage` | `acme-litellm-stage-gateway`       |
| `acme`   | `prod`  | `acme-litellm-prod-master-key`     |
| `globex` | `dev`   | `globex-litellm-dev-license`       |

For a per-tenant instance, the only inputs that change are the tenant
slug, env, and the two pre-issued secrets:

```bash
export TF_VAR_litellm_master_key="sk-..."   # the tenant's master key
export TF_VAR_litellm_license="lic-..."     # their LITELLM_LICENSE

terraform apply \
  -var "region=us-west-2" \
  -var 'azs=["us-west-2a","us-west-2b"]' \
  -var "tenant=acme" \
  -var "env=stage"
```

Both `litellm_master_key` and `litellm_license` are optional:
- Omit `litellm_master_key` → the stack auto-generates a random `sk-…`
  value (trial/dev path).
- Omit `litellm_license` → no license secret is created and gateway/
  backend run without `LITELLM_LICENSE` (OSS-only).

Use `TF_VAR_*` env vars rather than tfvars files for these — values
written to a tfvars file end up in `terraform.tfstate` and any committed
example files.

## Quick start

```bash
cd terraform/litellm/aws
cp terraform.tfvars.example terraform.tfvars
# Edit: region, tenant, env, azs, *_image, proxy_config, gateway_extra_secrets.

terraform init
terraform apply
```

That single apply provisions everything, runs the DB user bootstrap, runs the
schema migration, and only then starts the gateway/backend services. When it
returns, the stack is serving traffic.

```bash
terraform output alb_url
# UI login: admin / <master key>
aws secretsmanager get-secret-value \
  --secret-id "$(terraform output -raw master_key_secret_arn)" \
  --query SecretString --output text
```

## Image pulls

The defaults pull from `ghcr.io/berriai/litellm-<component>:v1.86.0-dev`,
which is anonymous-readable. There are four images: `litellm-gateway`,
`litellm-backend`, `litellm-ui`, and `litellm-migrations` (slim image used
only by the one-off migration task — runs `prisma migrate deploy` against
the writer DB and exits). Bump them together when bumping LiteLLM. To pull
from a private registry:

- **ECR (same account)**: the execution role already has
  `AmazonECSTaskExecutionRolePolicy`, which grants ECR pull for repos in
  the same account. No extra config needed.
- **ECR (cross-account)**: attach a policy to the execution role allowing
  `ecr:GetAuthorizationToken` + `ecr:BatchGetImage` on the foreign repo
  ARNs.
- **Other private registries** (GHCR with a PAT, Docker Hub, …): create a
  secret holding `{"auths":{"<registry>":{"auth":"<base64-user:token>"}}}`
  in Secrets Manager and set `repositoryCredentials.credentialsParameter`
  on the task def container — extend `ecs.tf` accordingly.

## TLS

`terraform plan` refuses to provision an HTTP-only ALB by default — TLS
is the supported posture. Two paths:

**Production / staging — provide an ACM certificate:**

1. Create or import an ACM cert in `var.region` covering the DNS name you
   plan to point at the ALB.
2. Set `acm_certificate_arn = "arn:aws:acm:..."` in tfvars and apply.

Result: a 443 listener carries the path-routing rules; the 80 listener
serves a permanent 301 redirect to HTTPS, so HTTP clients are
automatically upgraded.

**Trial / dev — explicitly opt into HTTP-only:**

Set `allow_plaintext_alb = true` in tfvars. Without this flag, plan fails
with a clear error pointing at the precondition. Intended for short-lived
trial / dev stacks only.

## Storage and database retention

Three opt-in tripwires guard against accidental data loss on
`terraform destroy`:

- **`skip_final_snapshot`** (Aurora; default `false`) — destroying the
  cluster takes a `<cluster>-final-<short-sha>` snapshot first.
- **`s3_force_destroy`** (S3 bucket holding request log archives,
  `/v1/files` content, and the S3 cache backend; default `false`) —
  `terraform destroy` against a non-empty bucket fails.

Flip either to `true` only for ephemeral / CI stacks where you accept
losing the contents.

## Files

| File              | What's in it                                                          |
| ----------------- | --------------------------------------------------------------------- |
| `versions.tf`     | Terraform + provider version constraints                              |
| `providers.tf`    | AWS provider (region + default tags)                                  |
| `variables.tf`    | All input variables                                                   |
| `locals.tf`       | Path-prefix lists for ALB routing (mirror of `helm/.../ingress.yaml`) |
| `network.tf`      | VPC, subnets, IGW, NAT, route tables, security groups                 |
| `secrets.tf`      | Secrets Manager entries + random passwords                            |
| `rds.tf`          | Aurora Postgres cluster + writer / reader instances                   |
| `redis.tf`        | ElastiCache Redis                                                     |
| `s3.tf`           | S3 bucket + task-role policy scoped to it                             |
| `iam.tf`          | Task execution + task roles, including `rds-db:connect`               |
| `ecs.tf`          | ECS cluster, task definitions, services for the three components     |
| `alb.tf`          | ALB, listener, target groups, path-routing rules                      |
| `migrations.tf`   | One-off migration task definition                                     |
| `outputs.tf`      | DNS name, secret ARN, bootstrap SQL, migration `run-task` command     |
