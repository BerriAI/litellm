variable "region" {
  description = "AWS region to deploy into."
  type        = string
}

variable "tenant" {
  description = "Tenant slug — used as the prefix for every AWS resource the stack creates. Combined with var.env to form `<tenant>-litellm-<env>` (e.g. `acme-litellm-stage`)."
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

variable "tags" {
  description = "Per-deployment tags applied to every taggable resource the module creates, on top of the module's own `litellm:stack` / `managed-by` tags. Caller-level provider `default_tags` (if any) merge with these."
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
    value is written to the master-key Secrets Manager entry. When empty,
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
    `<tenant>-litellm-<env>-license` Secrets Manager entry, grants the
    task-execution role GetSecretValue on it, and exposes its value to
    gateway + backend as `LITELLM_LICENSE`. Leave empty for OSS-only deploys.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "ui_password" {
  description = <<-EOT
    UI admin password. When set, the stack creates a
    `<tenant>-litellm-<env>-ui-password` Secrets Manager entry, grants the
    task-execution role GetSecretValue on it, and exposes its value to the
    backend as `UI_PASSWORD`. Pair with `backend_extra_env.UI_USERNAME` to
    set the matching username. Leave empty to skip — the proxy then falls
    back to the LITELLM_MASTER_KEY for UI login.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

# ---------- Networking ----------

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.40.0.0/16"
}

variable "azs" {
  description = "Availability zones to spread subnets across. At least 2 required for RDS and ALB."
  type        = list(string)
  validation {
    condition     = length(var.azs) >= 2
    error_message = "Provide at least 2 availability zones."
  }
}

# ---------- Component images ----------
#
# Defaults pin the four componentized images at the same release tag on
# GHCR. Override on a per-component basis in tfvars when bumping; bump them
# together when bumping the LiteLLM release.

variable "gateway_image" {
  description = "Container image for the gateway (data plane, port 4000). Tag must match a tag actually published to GHCR — the split images use the `v`-prefixed semver convention."
  type        = string
  default     = "ghcr.io/berriai/litellm-gateway:v1.86.0-dev"
}

variable "backend_image" {
  description = "Container image for the backend (management API, port 4001)."
  type        = string
  default     = "ghcr.io/berriai/litellm-backend:v1.86.0-dev"
}

variable "ui_image" {
  description = "Container image for the UI (nginx static export, port 3000)."
  type        = string
  default     = "ghcr.io/berriai/litellm-ui:v1.86.0-dev"
}

variable "migrations_image" {
  description = <<-EOT
    Container image for the one-off prisma migration task. Built from
    `migrations/Dockerfile` — slim image whose ENTRYPOINT runs
    `python3 /app/run.py` (assembles DATABASE_URL from DATABASE_* env vars
    via DatabaseURLSettings, then runs `prisma migrate deploy`). Should track
    the same release tag as gateway/backend/ui.
  EOT
  type        = string
  default     = "ghcr.io/berriai/litellm-migrations:v1.86.0-dev"
}

# ---------- Service sizing ----------

variable "gateway_cpu" {
  description = "Fargate CPU units for the gateway task (1024 = 1 vCPU)."
  type        = number
  default     = 1024
}

variable "gateway_memory" {
  description = "Fargate memory (MiB) for the gateway task."
  type        = number
  default     = 4096
}

variable "gateway_desired_count" {
  description = "Desired number of gateway tasks."
  type        = number
  default     = 2
}

variable "gateway_num_workers" {
  description = "uvicorn worker processes per gateway task (passed as --workers). Size relative to gateway_cpu — uvicorn recommends ~(2 × vCPU) + 1 for CPU-bound work."
  type        = number
  default     = 1

  validation {
    condition     = var.gateway_num_workers >= 1
    error_message = "gateway_num_workers must be >= 1."
  }
}

variable "backend_cpu" {
  description = "Fargate CPU units for the backend task (1024 = 1 vCPU)."
  type        = number
  default     = 1024
}

variable "backend_memory" {
  description = "Fargate memory (MiB) for the backend task. The proxy_server import chain alone needs >1 GiB; 4 GiB matches gateway."
  type        = number
  default     = 4096
}

variable "backend_desired_count" {
  description = "Desired number of backend tasks."
  type        = number
  default     = 1
}

variable "ui_cpu" {
  description = "Fargate CPU units for the UI task."
  type        = number
  default     = 256
}

variable "ui_memory" {
  description = "Fargate memory (MiB) for the UI task."
  type        = number
  default     = 512
}

variable "ui_desired_count" {
  description = "Desired number of UI tasks."
  type        = number
  default     = 1
}

# ---------- Autoscaling ----------
# Defaults mirror helm/litellm/values.yaml HPAs. The "*_desired_count" vars
# above seed the initial task count; once autoscaling is enabled, the service's
# desired_count is left to Application Auto Scaling (ecs.tf ignores future
# changes to it).

variable "gateway_autoscaling_enabled" {
  description = "Toggle Application Auto Scaling target-tracking on the gateway service."
  type        = bool
  default     = true
}

variable "gateway_min_capacity" {
  description = "Minimum gateway task count under autoscaling."
  type        = number
  default     = 1
}

variable "gateway_max_capacity" {
  description = "Maximum gateway task count under autoscaling."
  type        = number
  default     = 10
}

variable "gateway_cpu_target" {
  description = "Target average CPU utilization (%) for the gateway autoscaling policy."
  type        = number
  default     = 70
}

variable "gateway_memory_target" {
  description = "Target average memory utilization (%) for the gateway autoscaling policy. Set 0 to skip the memory policy and scale on CPU only."
  type        = number
  default     = 80
}

variable "backend_autoscaling_enabled" {
  description = "Toggle Application Auto Scaling target-tracking on the backend service."
  type        = bool
  default     = true
}

variable "backend_min_capacity" {
  description = "Minimum backend task count under autoscaling."
  type        = number
  default     = 1
}

variable "backend_max_capacity" {
  description = "Maximum backend task count under autoscaling."
  type        = number
  default     = 4
}

variable "backend_cpu_target" {
  description = "Target average CPU utilization (%) for the backend autoscaling policy."
  type        = number
  default     = 70
}

variable "ui_autoscaling_enabled" {
  description = "Toggle Application Auto Scaling target-tracking on the UI service. Off by default — UI is a static nginx export and one task is usually enough."
  type        = bool
  default     = false
}

variable "ui_min_capacity" {
  description = "Minimum UI task count under autoscaling."
  type        = number
  default     = 1
}

variable "ui_max_capacity" {
  description = "Maximum UI task count under autoscaling."
  type        = number
  default     = 3
}

variable "ui_cpu_target" {
  description = "Target average CPU utilization (%) for the UI autoscaling policy."
  type        = number
  default     = 80
}

# ---------- RDS ----------

variable "db_instance_class" {
  description = "Aurora instance class for both writer and reader."
  type        = string
  default     = "db.r6g.large"
}

variable "db_engine_version" {
  description = "Aurora Postgres engine version. Major version drives the parameter-group family (aurora-postgresql<major>)."
  type        = string
  default     = "16.4"
}

variable "db_name" {
  description = "Initial database name created on the Aurora cluster."
  type        = string
  default     = "litellm"
}

variable "db_master_username" {
  description = "Aurora master (superuser) username — used only to bootstrap the IAM-authed application user."
  type        = string
  default     = "postgres"
}

variable "db_username" {
  description = "IAM-authed Postgres user the proxy connects as. Must be CREATEd in the cluster and granted the rds_iam role — see terraform/litellm/aws/README.md."
  type        = string
  default     = "litellm_app"
}

# ---------- Redis ----------

variable "redis_node_type" {
  description = "ElastiCache node type."
  type        = string
  default     = "cache.t4g.small"
}

variable "redis_num_replicas" {
  description = "Number of read replicas in the Redis replication group. The primary plus this many replicas form the cluster — set to 0 for a single-node dev deployment, 1+ for HA. multi_az_enabled and automatic_failover_enabled require >= 1."
  type        = number
  default     = 1

  validation {
    condition     = var.redis_num_replicas >= 0
    error_message = "redis_num_replicas must be >= 0."
  }
}

# ---------- TLS ----------

variable "acm_certificate_arn" {
  description = <<-EOT
    ACM certificate ARN for the ALB's HTTPS listener. When set, the stack
    provisions a 443 listener carrying the same path-routing rules as the 80
    listener, and the 80 listener is rewritten to redirect HTTP→HTTPS. Leave
    empty ("") to disable TLS (must combine with `allow_plaintext_alb = true`
    for the plan to succeed — see README.md "TLS").
  EOT
  type        = string
  default     = ""
}

variable "allow_plaintext_alb" {
  description = <<-EOT
    Opt into HTTP-only mode on the ALB (port 80, no TLS). Default false:
    `terraform plan` fails when `acm_certificate_arn = ""` so the operator
    must either provide an ACM cert or consciously opt out. Intended for
    short-lived trial / dev stacks only.
  EOT
  type        = bool
  default     = false
}

# ---------- RDS ----------

variable "skip_final_snapshot" {
  description = "Skip the Aurora final snapshot on `terraform destroy`. Default false — destroying the cluster takes a snapshot first so data is recoverable. Set true only for ephemeral / CI environments where you accept permanent data loss on destroy."
  type        = bool
  default     = false
}

variable "s3_force_destroy" {
  description = <<-EOT
    Allow `terraform destroy` to delete the S3 bucket even when it still
    contains objects (request log archives, /v1/files storage, S3 cache
    backend). Default false — destroying a non-empty bucket fails, acting
    as a tripwire against accidental data loss. Set true only for
    ephemeral / CI environments. Mirrors the safety posture of
    `skip_final_snapshot` on Aurora.
  EOT
  type        = bool
  default     = false
}

# ---------- Extra env ----------

variable "gateway_extra_env" {
  description = <<-EOT
    Additional plain-text env vars for the gateway container. Use this for
    non-sensitive config (LANGFUSE_HOST, custom feature flags, …). For API
    keys, use gateway_extra_secrets instead.
  EOT
  type        = map(string)
  default     = {}
}

variable "backend_extra_env" {
  description = "Additional plain-text env vars for the backend container."
  type        = map(string)
  default     = {}
}

variable "gateway_extra_secrets" {
  description = <<-EOT
    Extra env vars sourced from AWS Secrets Manager. Map of env-var name to
    Secrets Manager ARN. Pass the bare secret ARN to inject the whole secret
    string as the env var value, or append ":<jsonKey>::" to extract a single
    JSON field (ECS docs).

    Example for OPENAI_API_KEY:
      gateway_extra_secrets = {
        OPENAI_API_KEY = "arn:aws:secretsmanager:us-west-2:111122223333:secret:openai-api-key-AbCdEf"
      }

    The stack's task execution role automatically gains GetSecretValue on every
    ARN referenced here (suffix-stripped).
  EOT
  type        = map(string)
  default     = {}
}

variable "backend_extra_secrets" {
  description = "Same shape as gateway_extra_secrets, but layered onto the backend container."
  type        = map(string)
  default     = {}
}

variable "proxy_config" {
  description = <<-EOT
    LiteLLM proxy config (the contents of config.yaml). Mirrors the helm
    chart's `gateway.config.proxy_config` value. Uploaded to S3 under
    `config/litellm-config.yaml` in the stack's bucket; gateway and backend
    container entrypoints download it to /tmp/litellm-config.yaml at task
    start (CONFIG_FILE_PATH is set automatically). The S3 object's etag is
    wired into the task definition, so editing this value produces a new
    task-def revision and a rolling redeploy.

    Example:
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
          master_key       = "os.environ/LITELLM_MASTER_KEY"
          database_url     = "os.environ/DATABASE_URL"
          ui_username      = "admin"
        }
      }

    Leave empty ({}) to skip mounting a config — the proxy then runs with
    defaults. Use the "os.environ/<NAME>" syntax in the YAML to reference
    env vars provided by *_extra_env or *_extra_secrets.
  EOT
  type        = any
  default     = {}
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the three services."
  type        = number
  default     = 30
}

# ---------- OpenTelemetry v2 ----------
#
# https://docs.litellm.ai/docs/observability/opentelemetry_v2
#
# OTel v2 is opt-in and gated entirely on otel_endpoint, matching the GCP
# stack. Leave otel_endpoint = "" and nothing OTel-related lands in the
# container env. Set it and the gateway and backend gain LITELLM_OTEL_V2=true
# plus the OTEL_* block (per-component OTEL_SERVICE_NAME, exporter, endpoint,
# environment name, capture-content), with OTEL_HEADERS sourced from
# otel_headers_secret_arn when provided.

variable "otel_endpoint" {
  description = <<-EOT
    OTLP collector endpoint (sets OTEL_ENDPOINT). Empty disables OTel
    entirely (no LITELLM_OTEL_V2, no OTEL_* env). Point at any
    OTLP-compatible backend (self-hosted collector, Grafana Tempo,
    Honeycomb, Datadog). Example: "http://otel-collector.internal:4318"
    for OTLP/HTTP.
  EOT
  type        = string
  default     = ""
}

variable "otel_exporter" {
  description = <<-EOT
    OTLP exporter protocol. One of "otlp_http", "otlp_grpc", or "console"
    (stdout, useful for verifying instrumentation against CloudWatch logs).
    Ignored when otel_endpoint is empty.
  EOT
  type        = string
  default     = "otlp_http"

  validation {
    condition     = contains(["otlp_http", "otlp_grpc", "console"], var.otel_exporter)
    error_message = "otel_exporter must be one of: otlp_http, otlp_grpc, console."
  }
}

variable "otel_environment_name" {
  description = <<-EOT
    Value for OTEL_ENVIRONMENT_NAME (becomes `deployment.environment` on
    every span). Defaults to var.env when empty so spans land tagged with
    the deployment env without extra wiring.
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

variable "otel_headers_secret_arn" {
  description = <<-EOT
    Secrets Manager ARN whose plaintext value becomes OTEL_HEADERS
    (comma-separated `key=value` pairs, typically used to pass an API key
    header to a managed collector). The execution role auto-gains
    secretsmanager:GetSecretValue on this ARN. Empty omits OTEL_HEADERS.
  EOT
  type        = string
  default     = ""
}
