variable "project" {
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
  description = "Resource labels merged into every label-supporting resource."
  type        = map(string)
  default = {
    "managed-by" = "terraform"
  }
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
    `gateway.config.proxy_config`. Passed to gateway, backend, and the
    migration job as a base64-encoded env var and decoded to
    /tmp/litellm-config.yaml at container start; CONFIG_FILE_PATH is set
    automatically. Reference env-injected secrets from the YAML via
    `os.environ/<NAME>`. Leave empty ({}) to skip.
  EOT
  type        = any
  default     = {}
}
