# LiteLLM on GCP (Cloud Run)

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://ssh.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FBerriAI%2Flitellm&cloudshell_workspace=terraform%2Flitellm%2Fgcp%2Fexamples%2Fdefault&cloudshell_tutorial=TUTORIAL.md&cloudshell_image=gcr.io/ds-artifacts-cloudshell/deploystack_custom_image&shellonly=true)

The button above opens the [DeployStack](https://github.com/GoogleCloudPlatform/deploystack) installer in Cloud Shell, walks you through `TUTORIAL.md`, and runs `terraform apply` once you've answered the prompts. The rest of this README is the manual / advanced path.

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

**Required override.** The `image_registry` default (`ghcr.io/berriai`)
does **not** work as-is — Cloud Run only accepts images from Artifact
Registry, `[region.]gcr.io`, or `docker.io`, and rejects `ghcr.io` URIs
at apply time. Every deploy (including HCP Terraform 1-click) must
supply either `image_registry` pointed at an Artifact Registry remote
repo backed by GHCR, or full per-component `*_image` URIs against
images you've already mirrored. The default is present only so
`terraform plan` succeeds during local iteration.

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
YAML-encoded and uploaded to a dedicated GCS bucket as `config.yaml`, then
mounted read-only into the gateway and backend at `/etc/litellm` via Cloud
Run v2's gcsfuse volume. `CONFIG_FILE_PATH` points at the mount path. A
hash of the YAML rides along as an env var so an edit to `proxy_config`
forces a new Cloud Run revision; without it the new file would sit in the
bucket unread until the next unrelated revision rollover. The migrations
job doesn't get the config (it only runs `prisma migrate deploy`).

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

### OpenTelemetry v2

OTel v2 (https://docs.litellm.ai/docs/observability/opentelemetry_v2) is
opt-in and gated entirely on `otel_endpoint`. Empty (default) and nothing
OTel-related lands in the container env. Set it and both gateway and
backend gain `LITELLM_OTEL_V2=true` plus the `OTEL_*` block, with
`OTEL_SERVICE_NAME` stamped per component (`${tenant}-litellm-${env}-gateway`
and `-backend`) so spans land tagged with the right hop. Any `OTEL_*` key
set in `gateway_extra_env` / `backend_extra_env` overrides the default for
that service (Cloud Run rejects duplicate env names, so the override is
predictable).

```hcl
otel_endpoint         = "https://otel.example.com:4318"
otel_exporter         = "otlp_http"  # or otlp_grpc
otel_environment_name = "prod"       # default: var.env
otel_headers_secret   = "projects/my-gcp-project/secrets/otel-headers"
```

`OTEL_HEADERS` is wired as a Secret Manager `secret_key_ref` since it
typically carries the collector's auth token; create the secret with the
literal header string, e.g. `Authorization=Bearer <token>`.

`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` defaults to
`no_content`; flip `otel_capture_message_content = "prompt_and_completion"`
only after auditing what lands in the backend, since prompts and
completions are typically sensitive.

Behavior matches the AWS stack 1:1; the only naming differences are
`otel_headers_secret` (a Secret Manager resource ID) vs AWS's
`otel_headers_secret_arn` (a Secrets Manager ARN).

### Enterprise billing metrics

License-gated request metering is opt-in and gated entirely on
`billing_metrics_endpoint`. Empty (default) and no billing env is added to
the container, so existing deployments are unchanged. Set it and both
gateway and backend export billable-request counts over OTLP/HTTP,
authenticating to the collector with the mTLS client certificate issued for
your deployment.

The proxy accepts the certificate, key, and CA bundle as either a file path
or literal PEM content. This stack takes the PEM, writes each one to its own
Secret Manager entry, grants the runtime service account
`roles/secretmanager.secretAccessor` on them, and injects them as Cloud Run
secret env vars `LITELLM_BILLING_METRICS_CLIENT_CERT` / `_CLIENT_KEY` (and
`_CA_CERT` when set), so no volume mount is needed.

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

Behavior matches the AWS stack 1:1; the variable names are identical

### Prometheus metrics sidecar

`gateway_metrics_port` adds a `metrics` sidecar
(`python -m litellm.proxy.prometheus_metrics_server`) to the gateway Cloud Run
service that aggregates the workers' samples over an in-memory volume shared
with the gateway container, so the collector's scrape never runs on an
inference worker. Cloud Run only routes traffic to the gateway container, so
the load balancer keeps hitting port 4000 (including the gateway's own
authenticated `/metrics`, which stays as it was) and the sidecar port is
reachable on localhost inside the instance only. To get the series out, the
stack also adds Google's
[Managed Service for Prometheus sidecar](https://cloud.google.com/stackdriver/docs/managed-prometheus/cloudrun-sidecar)
(`gateway_metrics_collector_image`) with a `RunMonitoring` config stored in
Secret Manager that scrapes `localhost:<port>/metrics` every 30s and writes to
Cloud Monitoring as `prometheus.googleapis.com/...` metrics. Enabling it grants
the runtime service account `roles/monitoring.metricWriter` and
`roles/logging.logWriter` on the project. Needs `gateway_image` v1.101.0 or
newer. See [Prometheus metrics](https://docs.litellm.ai/docs/proxy/prometheus)
for the metrics themselves

```hcl
gateway_metrics_port = 4001
```

The collector scrapes from inside the instance, so scrapes on an instance with
no in-flight requests can fail when CPU is throttled between requests. Keep
`gateway_min_instances` at 1 or more and, if you see gaps, enable
instance-based billing on the gateway service. Unlike the AWS stack there is
no `gateway_metrics_scrape_cidrs`: nothing outside the instance can reach the
sidecar port, so there is no network rule to open

### Autoscaling

Cloud Run scales the gateway on request concurrency (plus its built-in CPU
target), not on a metric you attach. Each instance takes up to
`gateway_max_instance_request_concurrency` requests at once (default 80)
and Cloud Run adds instances between `gateway_min_instances` and
`gateway_max_instances` when the in-flight count fills up. That is the
request-rate signal for this stack: lower the concurrency for LLM streams
that hold a worker for tens of seconds, since a stream counts as one request
for as long as it is open

There is no tokens-per-second path here. Cloud Run's autoscaler has no
custom-metric input, so the `litellm_total_tokens_metric_total` counter the
proxy exposes cannot drive it. If you need token-based scaling on GCP, run
the gateway on GKE with the Helm chart's `targetTokensPerSecond` (see
"Dependencies only" below) rather than wiring the counter into Cloud
Monitoring, which the autoscaler would ignore

### In-container connection pool

Each of the `gateway_num_workers` uvicorn workers opens its own Prisma pool
straight to Cloud SQL, so one instance holds `workers x connection_limit`
connections and the fleet's footprint against the database ceiling grows with
every instance Cloud Run adds. `gateway_connection_pool_enabled` runs a
PgBouncer (transaction mode, loopback) inside the gateway container that all
workers share, capping the instance at `gateway_pool_max_db_connections`
upstream connections however many workers it runs.
`gateway_pool_max_client_conn` bounds the worker-side connections the pooler
accepts. The module sets `LITELLM_PGBOUNCER_ENABLED`,
`LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS` and `LITELLM_PGBOUNCER_MAX_CLIENT_CONN`
on the gateway service only; the backend service and the migrations job keep
the direct connection

```hcl
gateway_num_workers             = 4
gateway_connection_pool_enabled = true
gateway_pool_max_db_connections = 20
gateway_pool_max_client_conn    = 1000
```

The pooler holds one static database password for the life of the instance.
This stack authenticates to Cloud SQL with the Secret Manager password (see
[Database authentication](#database-authentication)), so nothing else is
needed; a Cloud SQL Auth Proxy sidecar with IAM auth would not work with the
pool

The gateway container starts through `python -m gateway.launch` (the
componentized image's own entrypoint) rather than `uvicorn` directly. The
launcher reads these variables, starts the pooler once per instance before
uvicorn forks the workers and hands them its loopback `DATABASE_URL`. It also
honours `KEEPALIVE_TIMEOUT` from `gateway_extra_env` the way the image does

### Collector sidecar

`collector_enabled = true` adds a `spend-collector` container to the gateway
Cloud Run service that runs `python -m litellm.proxy.collector` from the gateway
image, and sets `LITELLM_COLLECTOR_ENABLED=true` on the gateway so its
uvicorn workers ship spend events (SpendLogs writes, key/team/user spend
updates, budget alerts) to the sidecar instead of running that pipeline in
the request path. This is the Terraform counterpart of helm's
`gateway.collector`. The default (`false`) leaves the service exactly as
before. It is independent of the metrics sidecars above, whose GMP scraper
already owns the `collector` container name.

Containers in one Cloud Run instance share localhost, so the sidecar listens
on loopback TCP (`tcp://127.0.0.1:${collector_port}`, default 4010)
instead of the Unix socket helm uses; the proxy rejects any non-loopback
address. The sidecar runs the same Redis CA + `DATABASE_URL` bootstrap as
the gateway container, gets the same database, Redis, master-key, license,
proxy config, and `gateway_extra_env` / `gateway_extra_secrets` values, and
runs with `LITELLM_JOB_ROLE=collector`. With `gateway_connection_pool_enabled`
it also gets the `LITELLM_PGBOUNCER_*` env, so its Prisma client goes through
the instance-local PgBouncer instead of opening a second pool straight to the
database. When it is unreachable the gateway falls back to in-process spend
tracking.

```hcl
collector_enabled = true
# collector_cpu               = "1000m"  # added on top of gateway_cpu
# collector_memory            = "2Gi"    # added on top of gateway_memory
# collector_buffer_size       = 1000
# collector_on_unavailable    = "fallback"  # or "drop"
# collector_drain_timeout_seconds = 10
```

Cloud Run allocates CPU per instance while requests are in flight, and the
sidecar shares that allocation. Spend events are shipped right after each
response, so this works with request-based billing, but keep
`gateway_min_instances >= 1` if spend must keep draining while an instance
is otherwise idle. Variable names match the AWS stack; only the resource
units differ (Cloud Run strings vs Fargate units)

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

For a per-tenant instance via the example root, the only inputs that
change are the tenant slug, env, and the two pre-issued secrets:

```bash
cd terraform/litellm/gcp/examples/default
export TF_VAR_litellm_master_key="sk-..."   # the tenant's master key
export TF_VAR_litellm_license="lic-..."     # their LITELLM_LICENSE

terraform apply \
  -var "project_id=my-gcp-project" \
  -var "region=us-central1" \
  -var "tenant=acme" \
  -var "env=stage"
```

To run *many* tenants from a single config, call the module with
`for_each` instead of one root per tenant — only possible because the
module declares no provider block (see "Using as a module").

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
cd terraform/litellm/gcp/examples/default
cp terraform.tfvars.example terraform.tfvars
# Edit: project, region, tenant, env, image_registry, proxy_config, gateway_extra_secrets.

terraform init
terraform apply
```

`examples/default/` is a thin root that configures the `google` /
`google-beta` providers and calls the module (`../../`). It exposes a
curated variable surface; for advanced knobs (per-component
CPU/memory/instances, Cloud SQL tier/edition, Memorystore tier,
per-component image pins) set them on the `module "litellm"` block in
`examples/default/main.tf`, or call the module from your own config — see
"Using as a module" below.

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

If the organization enforces
`constraints/compute.managed.requireOsConfig`, existing projects also
need VM Manager (OS Config) enabled in project metadata before apply.
The Serverless VPC Access connector creates GCE VMs; without this
metadata the connector operation fails, Terraform's error is vague,
and Cloud operation logs report an org-policy violation. New projects
created after the constraint is already on are usually fine. Existing
projects are not. Set it once per project:

```bash
gcloud compute project-info add-metadata \
  --project PROJECT_ID \
  --metadata=enable-osconfig=TRUE
```

Do not manage this from the module: project-wide metadata is often
owned by another Terraform root. Google's setup notes:
https://docs.cloud.google.com/compute/vm-manager/docs/setup#set_metadata_values

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

## Using as a module

The directory itself is a module with **no `provider` block** — the caller
owns provider config. You can call it directly with `for_each` (many
tenants from one config), `count`, `depends_on`, or providers configured
to impersonate a service account / target a different project:

```hcl
provider "google" {
  project = "my-gcp-project"
  region  = "us-central1"
}
provider "google-beta" {
  project = "my-gcp-project"
  region  = "us-central1"
}

module "litellm" {
  source = "github.com/BerriAI/litellm//terraform/litellm/gcp?ref=<tag>"

  project = "my-gcp-project"
  region  = "us-central1"
  tenant  = "acme"
  env     = "prod"
  # ...any of the inputs in variables.tf...
}
```

Both the default `google` and `google-beta` configs are inherited by the
module automatically through the call; declare both in the caller.

Labels: the module stamps its own `litellm-stack` and `managed-by` labels
onto every label-supporting resource (Cloud Run services and the
migrations job, Cloud SQL writer and reader, Memorystore, Secret Manager
entries, GCS buckets, the LB global address and forwarding rules) and
merges `var.labels` on top. Use the `labels` input for per-deployment
labels; mirrors the AWS stack's `tags` input.

**`for_each` shares one provider config.** The module's `versions.tf` declares
`google` / `google-beta` *without* `configuration_aliases`, so it only ever
receives the caller's single default (unaliased) `google` / `google-beta`
providers. That's deliberate — it keeps the one-command path simple — but it
means a `for_each` over the module runs every instance against the **same
project, region, and credentials**. Use `for_each` for many tenants in one
project (distinct `tenant`/`env`); it cannot fan out across projects or regions
on its own. To deploy into separate projects/regions, give each its own root
with its own provider config (one `examples/default`-style root per project),
or fork the module to add `configuration_aliases` and pass per-instance
`providers = { ... }`.

## Dependencies only (run LiteLLM on GKE)

Set `create_runtime = false` to provision Cloud SQL, Memorystore, GCS,
Secret Manager, and the runtime service account without Cloud Run or the
load balancer. For a Shared VPC, set the full host-project network ID and
skip PSA creation after the host project has configured it:

```hcl
create_runtime        = false
network_id            = "projects/<host>/global/networks/<vpc>"
create_psa_connection = false
```

The host project must already have Private Services Access configured on
that network and the Service Networking API enabled; the module cannot set
PSA up from a service project. GKE nodes must sit on the same Shared VPC so
the Cloud SQL and Memorystore private IPs are routable from the pods. Run
the root with its provider pointed at the project that should own the
dependencies. `create_runtime = true` with `network_id` set is also allowed,
but the Serverless VPC Access connector has to live in the same project as
the network, so that combination only works when the VPC is in the
deployment project

Map the outputs into the Helm values as follows:

```yaml
database:
  writer:
    host: <cloudsql_writer_ip>
    dbname: <db_name>
    passwordSecret:
      name: <kubernetes-secret-with-db-credentials>
  reader:
    host: <cloudsql_reader_ip>
    dbname: <db_name>
    passwordSecret:
      name: <kubernetes-secret-with-db-credentials>
redis:
  host: <redis_host>
  port: <redis_port>
masterKey:
  secretName: <kubernetes-secret-with-master-key>
```

Create the database Secret with keys `username` (the `db_username` output)
and `password` (read it with `gcloud secrets versions access latest
--secret=<db_password_secret_id>`), and the master key Secret from
`master_key_secret_id` the same way. Memorystore only accepts TLS by
default, so store the `redis_server_ca_pem` output in a third Secret,
mount it into the gateway and backend pods via `volumes` / `volumeMounts`,
and add `REDIS_SSL=true` and `REDIS_SSL_CA_CERTS=<mount path>` to each
component's `extraEnv`. Setting `redis_transit_encryption = false` removes
the CA plumbing at the cost of plaintext Redis traffic inside the VPC

The chart's pre-install/pre-upgrade migration hook runs the Prisma
migration, so nothing replaces the Cloud Run migrations Job in this mode

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

By default, Memorystore runs with
`transit_encryption_mode = "SERVER_AUTHENTICATION"`, so Cloud Run connects
via `rediss://`. The instance's self-signed CA cert
(`server_ca_certs[0].cert`) is shipped to gateway and backend as
`REDIS_CA_PEM_B64`; their entrypoint shell decodes it to
`/tmp/redis-ca.pem` before uvicorn starts and points `REDIS_SSL_CA_CERTS` at
that path. Set `redis_transit_encryption = false` to use plaintext Redis.
For GKE, use `redis_server_ca_pem` as described in the dependencies-only
section, or accept the security tradeoff of disabling transit encryption

## Files

| File              | What's in it                                                         |
| ----------------- | -------------------------------------------------------------------- |
| `versions.tf`     | Terraform + `required_providers` constraints (module declares no provider config) |
| `examples/default/` | Thin root: `google` / `google-beta` providers + a call to the module. The one-command deploy path. |
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
| `outputs.tf`      | LB IP, service URLs, dependency endpoints, secret IDs, migration command |
| `tests/`          | Plan-only mock-provider coverage for deployment modes and Redis encryption |
