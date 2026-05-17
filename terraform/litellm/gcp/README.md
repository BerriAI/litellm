# LiteLLM on GCP (Cloud Run)

Deploys the componentized LiteLLM proxy on GCP:

- **VPC** + Private Services Access range + a Serverless VPC Access connector
  so Cloud Run can reach private IPs
- **Cloud SQL for PostgreSQL** — primary instance + cross-zone read replica,
  password auth via Secret Manager
- **Memorystore (Redis)** for caching + rate limiting, private IP only
- **GCS bucket** — private, versioned, uniform IAM; exposed as `GCS_BUCKET_NAME`
- **Secret Manager** entries for `LITELLM_MASTER_KEY` and `DATABASE_PASSWORD`
- **Cloud Run v2** services for `gateway` (port 4000), `backend` (port 4001),
  and `ui` (port 3000), all using a shared runtime service account
- **Cloud Run Job** (`litellm-migrations`) that runs `prisma migrate deploy` from the dedicated `ghcr.io/berriai/litellm-migrations` image
- **External global HTTP(S) load balancer** with serverless NEGs and a URL
  map mirroring the helm-chart ingress path routing:
  - LLM data-plane prefixes → `gateway`
  - UI asset paths → `ui`
  - Everything else → `backend`

## Image pulls

There are four images: `litellm-gateway`, `litellm-backend`, `litellm-ui`,
and `litellm-migrations` (slim image used only by the one-off Cloud Run
Job — runs `prisma migrate deploy` against the writer DB and exits).
Bump them together when bumping LiteLLM.

Cloud Run only accepts images from Artifact Registry, `[region.]gcr.io`,
or `docker.io` — `ghcr.io` URIs are rejected at apply time. The four
images are published to GHCR upstream, so any real deploy needs an
Artifact Registry remote repository pointed at GHCR.

**One-time setup (per project):** create a remote repo and let Cloud Run
pull through it.

```bash
gcloud artifacts repositories create litellm \
  --repository-format=docker \
  --location=us-central1 \
  --mode=remote-repository \
  --remote-repo-config-desc="GitHub Container Registry passthrough" \
  --remote-docker-repo=https://ghcr.io
```

Then point the stack at it via `image_registry`:

```hcl
image_registry = "us-central1-docker.pkg.dev/my-gcp-project/litellm/berriai"
image_tag      = "v1.86.0-dev"
```

The four `litellm-<component>:${image_tag}` URIs are composed from those
two vars. Set `gateway_image` / `backend_image` / `ui_image` /
`migrations_image` only if you need a per-component override (custom
build, different tag).

Two further notes:

- The runtime SAs the stack creates do **not** need
  `roles/artifactregistry.reader` — Cloud Run pulls images using the
  per-project serverless agent
  (`service-<project-num>@serverless-robot-prod.iam.gserviceaccount.com`),
  not the runtime SA.
- For a fully air-gapped option, mirror the images into a regular AR
  repository instead of a remote repo:

  ```bash
  for c in gateway backend ui migrations; do
    docker pull ghcr.io/berriai/litellm-$c:<tag>
    docker tag  ghcr.io/berriai/litellm-$c:<tag> \
                us-central1-docker.pkg.dev/$PROJECT/litellm/$c:<tag>
    docker push us-central1-docker.pkg.dev/$PROJECT/litellm/$c:<tag>
  done
  ```

  then set `image_registry = "us-central1-docker.pkg.dev/$PROJECT/litellm"`
  (drop the `/berriai` suffix — the mirrored layout has no org segment).

## Database authentication

LiteLLM's `init_iam_db_url_from_env()` mints **AWS RDS** tokens via boto3 —
it doesn't speak GCP IAM. To IAM-auth against Cloud SQL from Cloud Run you'd
need the Cloud SQL Auth Proxy as a sidecar, which complicates the service
spec. This stack therefore uses **password authentication**:

- A random password is generated and stored in Secret Manager
  (`<name>-db-password`).
- Each Cloud Run service receives the password as `DATABASE_PASSWORD` via
  `value_source.secret_key_ref`.
- The container's entrypoint shim assembles `DATABASE_URL` (and
  `DATABASE_URL_READ_REPLICA`) from `DATABASE_HOST` / `DATABASE_PASSWORD`
  before exec'ing uvicorn — so the password never appears in the service
  spec or in logs.

If you need GCP-native IAM auth later, add `cloud-sql-proxy` as a sidecar
container under `template.template.containers` (Cloud Run v2 supports
multiple containers) and replace the password-based URL with the proxy's
Unix socket.

## Configuring the proxy

### `proxy_config`

Mirrors the helm chart's `gateway.config.proxy_config`. The map is
YAML-encoded and base64-passed to gateway, backend, and the migration job;
each container decodes it to `/tmp/litellm-config.yaml` at startup and sets
`CONFIG_FILE_PATH`.

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

LiteLLM resolves `os.environ/<NAME>` references against the container
environment. Provider API keys belong in `*_extra_secrets` and are
referenced from the YAML by env-var name.

### Extra env / secrets

Non-sensitive env vars:

```hcl
gateway_extra_env = {
  LANGFUSE_HOST = "https://us.cloud.langfuse.com"
}
```

Sensitive values — create the secret in Secret Manager first, then reference
its resource ID:

```bash
echo -n "sk-proj-..." | gcloud secrets create openai-api-key --data-file=-
```

```hcl
gateway_extra_secrets = {
  OPENAI_API_KEY = "projects/my-gcp-project/secrets/openai-api-key"
}
```

The Cloud Run runtime SA auto-gains `roles/secretmanager.secretAccessor` on
every secret referenced. **Pass the bare secret resource ID only** —
`projects/.../secrets/openai-api-key`, never the version-suffixed form
`projects/.../secrets/openai-api-key/versions/3`. The Cloud Run
`secret_key_ref` binding and the stack's IAM `secret_id` grant both
reject the version suffix; version is always resolved as `latest`. If
you need a pinned version, edit `local.gateway_extra_secret_kv` in
`cloudrun.tf` directly to set `version = "3"` for the entry in question.

## Tenant deployment

Every resource the stack creates is named `${tenant}-litellm-${env}` (or
that plus a per-resource suffix), so multiple tenants and multiple
environments coexist in the same project as long as the `(tenant, env)`
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
  -var "project=my-gcp-project" \
  -var "region=us-central1" \
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
cd terraform/litellm/gcp
cp terraform.tfvars.example terraform.tfvars
# Edit: project, region, tenant, env, *_image, proxy_config, gateway_extra_secrets.

terraform init
terraform apply
```

That single apply provisions everything, runs the prisma schema migration via
the Cloud Run job (auto-triggered by `bootstrap.tf`), and only then starts the
gateway/backend services. When it returns, the stack is serving traffic.

```bash
terraform output lb_url
# UI login: admin / <master key>
gcloud secrets versions access latest --secret="$(terraform output -raw master_key_secret_id)"
```

The `migration_run_command` output is preserved for break-glass manual re-runs.

**Prerequisite**: `gcloud` must be authenticated (`gcloud auth login`) and the
required APIs must be enabled (run, sqladmin, redis, secretmanager,
vpcaccess, compute, servicenetworking, storage, artifactregistry).

## TLS

`terraform plan` refuses to provision an HTTP-only LB by default — TLS
is the supported posture. Two paths:

**Production / staging — set `lb_domains`:**

1. `terraform apply` once with `allow_plaintext_lb = true` (intentional
   chicken-and-egg escape hatch) to provision the LB and read the anycast
   IP from `terraform output -raw lb_ip`.
2. Point each DNS name you want to serve from at that IP.
3. Set `lb_domains = ["proxy.example.com"]` and remove
   `allow_plaintext_lb`; re-apply.

Result: a 443 forwarding rule with a Google-managed cert covering each
listed domain; the 80 forwarding rule is rewritten to serve a permanent
301 redirect to HTTPS, so HTTP clients are automatically upgraded. The
managed cert sits in `PROVISIONING` for ~15-60 min on first apply until
DNS propagation completes — `gcloud compute ssl-certificates describe
<tenant>-litellm-<env>-cert` shows the state.

**Trial / dev — explicitly opt into HTTP-only:**

Set `allow_plaintext_lb = true` and leave `lb_domains = []`. Without the
flag, plan fails with a clear error pointing at the precondition.
Intended for short-lived trial / dev stacks only.

## Storage and database retention

Two opt-in tripwires guard against accidental data loss on
`terraform destroy`:

- **`cloudsql_deletion_protection`** (Cloud SQL writer + reader;
  default `true`) — destroy fails with a clear error rather than
  dropping the database.
- **`gcs_force_destroy`** (GCS bucket holding request log archives,
  `/v1/files` content, and the GCS cache backend; default `false`) —
  `terraform destroy` against a non-empty bucket fails.

Flip `cloudsql_deletion_protection` to `false` or `gcs_force_destroy` to
`true` only for ephemeral / CI stacks where you accept losing the data.

## Redis encryption

Memorystore runs with `transit_encryption_mode = "SERVER_AUTHENTICATION"`,
so the proxy connects via `rediss://`. The instance's self-signed CA cert
(`server_ca_certs[0].cert`) is shipped to gateway + backend as
`REDIS_CA_PEM_B64`; their entrypoint shell decodes it to `/tmp/redis-ca.pem`
before uvicorn starts and points `REDIS_SSL_CA_CERTS` at that path. No
extra config needed — but if you ever swap Memorystore for an external
Redis, override `REDIS_HOST`/`REDIS_PORT` and either drop these env vars
or point them at your own CA.

## Files

| File              | What's in it                                                         |
| ----------------- | -------------------------------------------------------------------- |
| `versions.tf`     | Terraform + provider version constraints                             |
| `providers.tf`    | Google + Google-Beta providers                                       |
| `variables.tf`    | All input variables                                                  |
| `locals.tf`       | Path-prefix lists (mirror of `helm/.../ingress.yaml`) + proxy_config helpers |
| `network.tf`      | VPC, subnet, PSA range, Serverless VPC connector                     |
| `secrets.tf`      | Secret Manager entries + random master_key                           |
| `cloudsql.tf`     | Cloud SQL writer + read replica + app user + password secret         |
| `redis.tf`        | Memorystore Redis (private IP)                                       |
| `gcs.tf`          | GCS bucket + objectAdmin binding                                     |
| `iam.tf`          | Runtime SA + Cloud SQL client + Secret Manager accessor              |
| `cloudrun.tf`     | 3 Cloud Run services + Cloud Run Job for migrations                  |
| `load_balancer.tf`| External HTTPS LB, serverless NEGs, URL map for path routing         |
| `outputs.tf`      | LB IP, service URLs, secret IDs, migration `execute` command         |
