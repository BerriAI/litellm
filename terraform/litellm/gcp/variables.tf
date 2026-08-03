variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region for VPC, Cloud SQL, Memorystore, Cloud Run, and the LB IP."
  type        = string
  default     = "us-central1"
}

variable "tenant" {
  description = "Tenant slug — used as the prefix for every GCP resource the stack creates. Combined with var.env to form `<tenant>-litellm-<env>` (e.g. `acme-litellm-stage`)."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,20}$", var.tenant))
    error_message = "tenant must be 1-21 chars, lower-kebab-case, starting with a letter."
  }
}

variable "env" {
  description = "Environment suffix appended to every resource name (e.g. `stage`, `prod`, `dev`)."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,8}$", var.env))
    error_message = "env must be 1-9 chars, lower-kebab-case, starting with a letter."
  }
}

variable "labels" {
  description = "Per-deployment labels applied to every label-supporting resource the module creates, on top of the module's own `litellm-stack` / `managed-by` labels. Mirrors the AWS stack's `tags` input."
  type        = map(string)
  default     = {}
}

# ---------- Tenant-supplied secrets ----------
#
# Both default to "" so the stack stays usable for trial / OSS deploys.
# Set via TF_VAR_litellm_master_key / TF_VAR_litellm_license to keep the
# values out of state files committed to a VCS.

variable "litellm_master_key" {
  description = <<-EOT
    Pre-existing LITELLM_MASTER_KEY (must begin with `sk-`). When set, this
    value is written to the master-key Secret Manager entry. When empty,
    the stack auto-generates a random `sk-…` key (preserving today's
    trial-deploy behavior).
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "litellm_license" {
  description = <<-EOT
    LiteLLM enterprise license string. When set, the stack creates a
    `<tenant>-litellm-<env>-license` Secret Manager entry, grants the
    runtime SA accessor on it, and exposes its value to gateway + backend
    as `LITELLM_LICENSE`. Leave empty for OSS-only deploys.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "ui_password" {
  description = <<-EOT
    UI admin password. When set, the stack creates a
    `<tenant>-litellm-<env>-ui-password` Secret Manager entry, grants the
    runtime SA accessor on it, and exposes its value to the backend as
    `UI_PASSWORD`. Pair with `backend_extra_env.UI_USERNAME` to set the
    matching username. Leave empty to skip — the proxy then falls back to
    the LITELLM_MASTER_KEY for UI login.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

# ---------- Networking ----------

variable "subnet_cidr" {
  description = "Primary CIDR block for the LiteLLM subnet."
  type        = string
  default     = "10.40.0.0/16"
}

variable "vpc_connector_cidr" {
  description = "CIDR for the Serverless VPC Access connector. /28 required."
  type        = string
  default     = "10.41.0.0/28"
}

# ---------- Component images ----------
#
# Cloud Run only pulls from Artifact Registry, [region.]gcr.io, or
# docker.io — it rejects arbitrary registries (notably ghcr.io) at apply
# time. The four images live on GHCR upstream, so any real deploy must
# either set `image_registry` to an Artifact Registry remote repository
# pointed at ghcr.io (e.g. `us-central1-docker.pkg.dev/my-proj/litellm/berriai`)
# or override the per-component `*_image` vars individually with full URIs.

variable "image_registry" {
  description = <<-EOT
    Registry path prefix used to compose the four LiteLLM image URIs as
    `<image_registry>/litellm-<component>:<image_tag>`. The default
    (`ghcr.io/berriai`) only works on registries Cloud Run accepts — for
    GHCR-backed deploys, create an Artifact Registry remote repository
    pointed at `https://ghcr.io` and set this to that repo's path
    (e.g. `us-central1-docker.pkg.dev/<project>/<remote-repo>/berriai`).
    Per-component overrides (`gateway_image`, `backend_image`, `ui_image`,
    `migrations_image`) bypass this entirely when set.
  EOT
  type        = string
  default     = "ghcr.io/berriai"
}

variable "image_tag" {
  description = "Tag applied to all four litellm-* images when composed from `image_registry`. Bump in lockstep when bumping LiteLLM. Must match a tag actually published to GHCR — the split images use the `v`-prefixed semver convention (e.g. `v1.86.0-dev`)."
  type        = string
  default     = "v1.86.0-dev"
}

variable "gateway_image" {
  description = "Full image URI for the gateway. Empty (default) composes from `image_registry` + `image_tag`. Public images or Artifact Registry only — Cloud Run won't authenticate against arbitrary private registries."
  type        = string
  default     = ""
}

variable "backend_image" {
  description = "Full image URI for the backend. Empty (default) composes from `image_registry` + `image_tag`."
  type        = string
  default     = ""
}

variable "ui_image" {
  description = "Full image URI for the UI. Empty (default) composes from `image_registry` + `image_tag`."
  type        = string
  default     = ""
}

variable "migrations_image" {
  description = <<-EOT
    Full image URI for the one-off prisma migration Cloud Run Job. Empty
    (default) composes from `image_registry` + `image_tag` as
    `litellm-migrations`. Built from `migrations/Dockerfile` — slim image
    whose ENTRYPOINT runs `python3 /app/run.py` (assembles DATABASE_URL
    from DATABASE_* env vars via DatabaseURLSettings, then runs
    `prisma migrate deploy`). Should track the same release tag as
    gateway/backend/ui.
  EOT
  type        = string
  default     = ""
}

# ---------- Service sizing ----------

variable "gateway_cpu" {
  description = "Cloud Run CPU per gateway instance."
  type        = string
  default     = "1000m"
}

variable "gateway_memory" {
  description = "Cloud Run memory per gateway instance."
  type        = string
  default     = "4Gi"
}

variable "gateway_num_workers" {
  description = "uvicorn worker processes per gateway instance (passed as --workers). Size relative to gateway_cpu — uvicorn recommends ~(2 × vCPU) + 1 for CPU-bound work. Mirrors the AWS stack's gateway_num_workers."
  type        = number
  default     = 1

  validation {
    condition     = var.gateway_num_workers >= 1
    error_message = "gateway_num_workers must be >= 1."
  }
}

# Cloud Run autoscales out of the box (request-rate driven). The min/max
# bounds mirror the HPA replica bounds in helm/litellm/values.yaml so each
# stack scales over the same range. Cloud Run has no direct CPU-utilization
# target; the request-concurrency knob below is the closest analog.

variable "gateway_min_instances" {
  description = "Lower bound on gateway Cloud Run instances. Matches helm HPA minReplicas."
  type        = number
  default     = 1
}

variable "gateway_max_instances" {
  description = "Upper bound on gateway Cloud Run instances. Matches helm HPA maxReplicas."
  type        = number
  default     = 10
}

variable "gateway_max_instance_request_concurrency" {
  description = "Concurrent requests one gateway instance handles before Cloud Run scales out. Cloud Run v2 default is 80; lower it for LLM streams that pin a worker for tens of seconds."
  type        = number
  default     = 80
}

variable "backend_cpu" {
  description = "Cloud Run CPU per backend instance. Cloud Run rejects sub-1 CPU when `backend_max_instance_request_concurrency > 1`, so the default is 1000m. Lower this only if you also drop concurrency to 1."
  type        = string
  default     = "1000m"
}

variable "backend_memory" {
  description = "Cloud Run memory per backend instance."
  type        = string
  default     = "4Gi"
}

variable "backend_min_instances" {
  description = "Lower bound on backend Cloud Run instances. Matches helm HPA minReplicas."
  type        = number
  default     = 1
}

variable "backend_max_instances" {
  description = "Upper bound on backend Cloud Run instances. Matches helm HPA maxReplicas."
  type        = number
  default     = 4
}

variable "backend_max_instance_request_concurrency" {
  description = "Concurrent requests one backend instance handles before Cloud Run scales out."
  type        = number
  default     = 80
}

variable "ui_cpu" {
  description = "Cloud Run CPU per UI instance. Cloud Run rejects sub-1 CPU when `ui_max_instance_request_concurrency > 1`, so the default is 1000m. Lower this only if you also drop concurrency to 1 (which makes nginx scale 1:1 with traffic — almost never what you want)."
  type        = string
  default     = "1000m"
}

variable "ui_memory" {
  description = "Cloud Run memory per UI instance. Cloud Run rejects `< 512Mi` when CPU is always-allocated (the default whenever `ui_min_instances > 0`), so the default is 512Mi."
  type        = string
  default     = "512Mi"
}

variable "ui_min_instances" {
  description = "Lower bound on UI Cloud Run instances. Matches helm HPA minReplicas."
  type        = number
  default     = 1
}

variable "ui_max_instances" {
  description = "Upper bound on UI Cloud Run instances. Matches helm HPA maxReplicas."
  type        = number
  default     = 3
}

variable "ui_max_instance_request_concurrency" {
  description = "Concurrent requests one UI instance handles before Cloud Run scales out. The UI is static nginx, so this can be high."
  type        = number
  default     = 200
}

# ---------- Cloud SQL ----------

variable "db_tier" {
  description = "Cloud SQL tier (machine type) for the writer instance."
  type        = string
  default     = "db-custom-2-7680"
}

variable "db_edition" {
  description = "Cloud SQL edition. ENTERPRISE accepts the db-custom-* and db-n1-* tiers. ENTERPRISE_PLUS only accepts db-perf-optimized-* tiers and is ~3x cost — change db_tier in lockstep when switching."
  type        = string
  default     = "ENTERPRISE"

  validation {
    condition     = contains(["ENTERPRISE", "ENTERPRISE_PLUS"], var.db_edition)
    error_message = "db_edition must be ENTERPRISE or ENTERPRISE_PLUS."
  }
}

variable "db_version" {
  description = "Cloud SQL Postgres version."
  type        = string
  default     = "POSTGRES_16"
}

variable "db_name" {
  description = "Initial database created on the Cloud SQL instance."
  type        = string
  default     = "litellm"
}

variable "db_username" {
  description = "Application Postgres user (password-auth). Password is auto-generated and stored in Secret Manager."
  type        = string
  default     = "litellm_app"
}

variable "lb_domains" {
  description = <<-EOT
    DNS names for a Google-managed SSL certificate fronting the LB. When
    non-empty, the stack provisions a 443 forwarding rule + HTTPS target
    proxy + managed cert covering these domains, and the existing 80
    forwarding rule serves a permanent 301 redirect to HTTPS. Leave empty
    ([]) to disable TLS (must combine with `allow_plaintext_lb = true` for
    the plan to succeed — see README.md "TLS"). Each domain must already
    resolve to the LB's anycast IP (`lb_ip` output) for managed-cert
    provisioning to succeed.
  EOT
  type        = list(string)
  default     = []
}

variable "allow_plaintext_lb" {
  description = <<-EOT
    Opt into HTTP-only mode on the load balancer (port 80, no TLS).
    Default false: `terraform plan` fails when `lb_domains = []` so the
    operator must either provide DNS names for a managed cert or
    consciously opt out. Intended for short-lived trial / dev stacks only.
  EOT
  type        = bool
  default     = false
}

variable "cloudsql_deletion_protection" {
  description = "Cloud SQL instance-level deletion protection (writer + reader). Default true — `terraform destroy` (and `terraform apply` operations that replace the instance) will fail with a clear error rather than silently dropping the database. Set false only for ephemeral / CI environments."
  type        = bool
  default     = true
}

variable "gcs_force_destroy" {
  description = <<-EOT
    Allow `terraform destroy` to delete the GCS bucket even when it still
    contains objects (request log archives, /v1/files storage, GCS cache
    backend). Default false — destroying a non-empty bucket fails, acting
    as a tripwire against accidental data loss. Set true only for
    ephemeral / CI environments. Mirrors `s3_force_destroy` on AWS and
    `cloudsql_deletion_protection` on the database side.
  EOT
  type        = bool
  default     = false
}

# ---------- Memorystore (Redis) ----------

variable "redis_tier" {
  description = "Memorystore tier — STANDARD_HA for production, BASIC for dev."
  type        = string
  default     = "STANDARD_HA"
}

variable "redis_memory_size_gb" {
  type    = number
  default = 1
}

# ---------- Extras / proxy_config ----------

variable "gateway_extra_env" {
  description = "Plain-text env vars layered onto the gateway."
  type        = map(string)
  default     = {}
}

variable "backend_extra_env" {
  description = "Plain-text env vars layered onto the backend."
  type        = map(string)
  default     = {}
}

variable "gateway_extra_secrets" {
  description = <<-EOT
    Extra env vars sourced from Google Secret Manager, applied to the
    gateway. Map of env-var name to the Secret Manager **secret resource
    ID** (`projects/<project>/secrets/<name>` — *not* a version resource
    ID; the Cloud Run secret_key_ref binding and the stack's IAM grant
    both reject `/versions/<n>` suffixes). Versions are always resolved
    as `latest`; if you need a pinned version, edit
    `local.gateway_extra_secret_kv` in `cloudrun.tf` directly.

    Example:
      gateway_extra_secrets = {
        OPENAI_API_KEY = "projects/my-proj/secrets/openai-api-key"
      }

    The Cloud Run service account auto-gains roles/secretmanager.secretAccessor
    on each secret listed here.
  EOT
  type        = map(string)
  default     = {}
}

variable "backend_extra_secrets" {
  description = "Same shape as gateway_extra_secrets (secret resource ID, version always `latest`), layered onto the backend."
  type        = map(string)
  default     = {}
}

variable "proxy_config" {
  description = <<-EOT
    LiteLLM proxy config (contents of config.yaml). Mirrors the helm chart's
    `gateway.config.proxy_config`. YAML-encoded and uploaded to a dedicated
    GCS bucket as `config.yaml`, then mounted read-only into the gateway
    and backend at `/etc/litellm` via Cloud Run v2's gcsfuse volume;
    CONFIG_FILE_PATH is set automatically. A hash of the YAML is wired in
    as an env var so a config-only edit forces a new revision (gcsfuse
    surfaces the new object on container restart). Reference env-injected
    secrets from the YAML via `os.environ/<NAME>`. Leave empty ({}) to
    skip — the bucket isn't created and no volume is mounted.
  EOT
  type        = any
  default     = {}
}

# ---------- OpenTelemetry v2 ----------
#
# https://docs.litellm.ai/docs/observability/opentelemetry_v2
#
# OTel v2 is opt-in and gated entirely on otel_endpoint, matching the AWS
# stack. Leave otel_endpoint = "" and nothing OTel-related is added to the
# container env. Set it and the gateway/backend gain LITELLM_OTEL_V2=true
# plus the OTEL_* block (per-component OTEL_SERVICE_NAME, exporter, endpoint,
# environment name, capture-content), with OTEL_HEADERS sourced from
# otel_headers_secret when provided.

variable "otel_endpoint" {
  description = <<-EOT
    OTLP collector URL (e.g. https://otel.example.com:4318 for HTTP, or
    your collector's :4317 for gRPC). Empty disables OTel entirely (no
    LITELLM_OTEL_V2, no OTEL_* env). When set, LITELLM_OTEL_V2=true plus
    OTEL_EXPORTER / OTEL_ENDPOINT are injected and spans ship to the
    collector.
  EOT
  type        = string
  default     = ""
}

variable "otel_exporter" {
  description = <<-EOT
    OTel exporter protocol. Ignored when otel_endpoint is empty. `otlp_http`
    is the safer default (works through a vanilla L7 ingress); `otlp_grpc`
    needs the collector reachable over h2 and the `grpcio` extra installed
    in the proxy image.
  EOT
  type        = string
  default     = "otlp_http"
  validation {
    condition     = contains(["otlp_http", "otlp_grpc", "console"], var.otel_exporter)
    error_message = "otel_exporter must be one of: otlp_http, otlp_grpc, console."
  }
}

variable "otel_headers_secret" {
  description = <<-EOT
    Optional Secret Manager secret resource ID
    (`projects/<project>/secrets/<name>`) whose latest version is the
    value of OTEL_HEADERS — used for collector auth, e.g.
    `Authorization=Bearer <token>`. Mounted as an env-var secret_key_ref;
    the runtime SA auto-gains roles/secretmanager.secretAccessor.
  EOT
  type        = string
  default     = ""
}

variable "otel_environment_name" {
  description = <<-EOT
    Value for OTEL_ENVIRONMENT_NAME (becomes `deployment.environment` on
    every span). Defaults to var.env so spans land tagged with the
    deployment env without extra wiring.
  EOT
  type        = string
  default     = ""
}

variable "otel_capture_message_content" {
  description = <<-EOT
    Value for OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT. Default
    `no_content` matches the litellm default; flip to `prompt_and_completion`
    only when you've audited what's about to land in your observability
    backend, because raw prompts/completions are typically sensitive.
  EOT
  type        = string
  default     = "no_content"
  validation {
    condition     = contains(["no_content", "prompt_and_completion"], var.otel_capture_message_content)
    error_message = "otel_capture_message_content must be one of: no_content, prompt_and_completion."
  }
}

# ---------- Enterprise billing metrics ----------
#
# License-gated request metering. Opt-in and gated entirely on
# billing_metrics_endpoint: leave it empty (the default) and nothing
# metering-related is added to the container env. Set it and gateway +
# backend export billable-request counts over OTLP/HTTP, authenticating to
# the collector with an mTLS client cert. The proxy accepts the cert, key,
# and CA as either a file path or literal PEM content, so on Cloud Run they
# are injected straight from Secret Manager as env vars and no volume is
# needed.

variable "billing_metrics_endpoint" {
  description = <<-EOT
    OTLP/HTTP endpoint for enterprise billing metrics (sets
    LITELLM_BILLING_METRICS_ENDPOINT). Non-empty enables request metering;
    empty (default) disables it and adds no billing env to the container.
    Requires an enterprise license. Example:
    "https://telemetry.litellm.ai/v1/metrics"
  EOT
  type        = string
  default     = ""
}

variable "billing_metrics_client_cert_pem" {
  description = <<-EOT
    PEM content of the mTLS client certificate issued for this deployment.
    When billing_metrics_endpoint is set, the stack stores this in a
    `<tenant>-litellm-<env>-billing-metrics-client-cert` Secret Manager
    entry, grants the runtime SA accessor on it, and exposes it to gateway +
    backend as LITELLM_BILLING_METRICS_CLIENT_CERT. Required whenever
    metering is enabled.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "billing_metrics_client_key_pem" {
  description = <<-EOT
    PEM content of the private key matching
    billing_metrics_client_cert_pem. Stored in a
    `<tenant>-litellm-<env>-billing-metrics-client-key` Secret Manager entry
    and exposed as LITELLM_BILLING_METRICS_CLIENT_KEY. Required whenever
    metering is enabled.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "billing_metrics_ca_cert_pem" {
  description = <<-EOT
    PEM content of the CA bundle used to verify the metering collector.
    Only needed for private or test collectors whose CA is not in the
    system trust store; telemetry.litellm.ai is publicly trusted, so leave
    this empty for production. When set, it is exposed as
    LITELLM_BILLING_METRICS_CA_CERT.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}
