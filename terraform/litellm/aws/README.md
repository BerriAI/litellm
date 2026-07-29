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
and uploaded to S3 (`config/litellm-config.yaml` in the stack's bucket); the
gateway and backend container entrypoints download it to
`/tmp/litellm-config.yaml` at task start via boto3 and set `CONFIG_FILE_PATH`
to match. The S3 object's etag is wired into the task definition, so editing
`proxy_config` produces a new task-def revision and a rolling redeploy of both
services.

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

### Observability (OpenTelemetry v2)

OTel v2 (https://docs.litellm.ai/docs/observability/opentelemetry_v2) is
opt-in and gated entirely on `otel_endpoint`. Empty (default) and nothing
OTel-related is added to the container env. Set it and both gateway and
backend gain `LITELLM_OTEL_V2=true` plus the `OTEL_*` block, with
`OTEL_SERVICE_NAME` stamped per component (`${tenant}-litellm-${env}-gateway`
and `-backend`) so spans land tagged with the right hop. Any `OTEL_*` key
set in `gateway_extra_env` / `backend_extra_env` overrides the default for
that service.

```hcl
otel_endpoint         = "http://otel-collector.internal:4318"
otel_exporter         = "otlp_http"   # otlp_grpc, console
otel_environment_name = "prod"        # defaults to var.env
```

For collectors that require an auth header, store the comma-separated
`key=value` string in Secrets Manager and reference it via
`otel_headers_secret_arn`. The execution role auto-gains
`secretsmanager:GetSecretValue` on that ARN.

```hcl
otel_headers_secret_arn = "arn:aws:secretsmanager:us-west-2:111122223333:secret:honeycomb-otel-headers-AbCdEf"
```

`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` defaults to
`no_content`; flip `otel_capture_message_content = "prompt_and_completion"`
only after auditing what lands in the backend, since prompts and
completions are typically sensitive.

Vendor presets (Arize, Phoenix, Langfuse OTel, Weave, Langtrace, Levo,
AgentOps) live under `proxy_config.litellm_settings.callbacks` and are
orthogonal to the OTLP variables above; their credentials still go in
`*_extra_secrets`.

### Enterprise billing metrics

License-gated request metering is opt-in and gated entirely on
`billing_metrics_endpoint`. Empty (default) and no billing env is added to
the container, so existing deployments are unchanged. Set it and both
gateway and backend export billable-request counts over OTLP/HTTP,
authenticating to the collector with the mTLS client certificate issued for
your deployment.

The proxy accepts the certificate, key, and CA bundle as either a file path
or literal PEM content. This stack takes the PEM, writes each one to its own
Secrets Manager entry, grants the task-execution role
`secretsmanager:GetSecretValue` on them, and injects them as
`LITELLM_BILLING_METRICS_CLIENT_CERT` / `_CLIENT_KEY` (and `_CA_CERT` when
set), so no volume mount is needed on Fargate.

```hcl
billing_metrics_endpoint = "https://telemetry.litellm.ai/v1/metrics"
```

```bash
export TF_VAR_billing_metrics_client_cert_pem="$(cat client.crt)"
export TF_VAR_billing_metrics_client_key_pem="$(cat client.key)"
```

`billing_metrics_ca_cert_pem` is only for private or test collectors whose
CA is not in the system trust store; leave it empty against
`telemetry.litellm.ai`. Metering requires an enterprise license, so pair
this with `litellm_license`. To tune the export cadence, set
`LITELLM_BILLING_METRICS_EXPORT_INTERVAL_MS` through `gateway_extra_env` /
`backend_extra_env`

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

For a per-tenant instance via the example root, the only inputs that
change are the tenant slug, env, and the two pre-issued secrets:

```bash
cd terraform/litellm/aws/examples/default
export TF_VAR_litellm_master_key="sk-..."   # the tenant's master key
export TF_VAR_litellm_license="lic-..."     # their LITELLM_LICENSE

terraform apply \
  -var "region=us-west-2" \
  -var 'azs=["us-west-2a","us-west-2b"]' \
  -var "tenant=acme" \
  -var "env=stage"
```

To run *many* tenants from a single config, call the module with
`for_each` instead of one root per tenant (see "Using as a module"):

```hcl
module "litellm" {
  for_each = toset(["acme", "globex"])
  source   = "github.com/BerriAI/litellm//terraform/litellm/aws?ref=<tag>"
  tenant   = each.key
  env      = "prod"
  region   = "us-west-2"
  azs      = ["us-west-2a", "us-west-2b"]
}
```
(This `for_each` form is only possible because the module declares no
provider block — the original root-with-provider layout forbade it.)

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
cd terraform/litellm/aws/examples/default
cp terraform.tfvars.example terraform.tfvars
# Edit: region, tenant, env, azs, proxy_config, gateway_extra_secrets.

terraform init
terraform apply
```

`examples/default/` is a thin root that configures the `aws` provider and
calls the module (`../../`). It exposes a curated variable surface; for
advanced knobs (per-component CPU/memory/workers, autoscaling, RDS/Redis
sizing, per-component image pins) set them on the `module "litellm"` block
in `examples/default/main.tf`, or call the module from your own config —
see "Using as a module" below.

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

## Using as a module

The directory itself is a module with **no `provider` block** — the caller
owns provider config. That means you can call it directly with `for_each`
(many tenants from one config), `count` (conditional stacks), `depends_on`,
an assume-role / aliased provider, etc.:

```hcl
provider "aws" {
  region = "us-west-2"
  assume_role { role_arn = "arn:aws:iam::111122223333:role/deployer" }
}

module "litellm" {
  source = "github.com/BerriAI/litellm//terraform/litellm/aws?ref=<tag>"

  region = "us-west-2"
  tenant = "acme"
  env    = "prod"
  azs    = ["us-west-2a", "us-west-2b"]
  # ...any of the inputs in variables.tf...
}
```

Tags: the module threads its own `litellm:stack` / `managed-by` / `var.tags`
onto every taggable resource. Any `default_tags` on your provider merge on
top — set org-wide tags there, per-deployment tags via the `tags` input.

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
| `versions.tf`     | Terraform + `required_providers` constraints (module declares no provider config) |
| `examples/default/` | Thin root: `aws` provider (with an optional `default_tags` slot for org-wide tags) + a call to the module. The one-command deploy path. |
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
