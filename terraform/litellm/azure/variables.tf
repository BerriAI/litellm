variable "location" {
  description = "Azure region to deploy into (e.g. `eastus`, `westus2`)."
  type        = string
}

variable "tenant" {
  description = "Tenant slug, used as the prefix for every Azure resource the stack creates. Combined with var.env to form `<tenant>-litellm-<env>` (e.g. `acme-litellm-stage`)."
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
    error_message = "env must be 1-9 chars, lower-kebab-case, starting a letter."
  }
}

variable "tags" {
  description = "Per-deployment tags applied to every taggable resource the module creates, on top of the module's own `litellm:stack` / `managed-by` tags. Caller-level provider default_tags (if any) merge with these."
  type        = map(string)
  default     = {}
}

variable "resource_group_name" {
  description = "Existing resource group to deploy into. Leave empty to have the module create one named `<tenant>-litellm-<env>-rg`."
  type        = string
  default     = ""
}

# ---------- Tenant-supplied secrets ----------
#
# Both default to "" so the stack stays usable for trial / OSS deploys.
# Set via TF_VAR_litellm_master_key / TF_VAR_litellm_license to keep the
# values out of state files committed to a VCS.

variable "litellm_master_key" {
  description = <<-EOT
    Pre-existing LITELLM_MASTER_KEY (must begin with `sk-`). When set, this
    value is written to the master-key Key Vault entry. When empty, the
    stack auto-generates a random `sk-...` key (preserving today's
    trial-deploy behavior).
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "litellm_license" {
  description = <<-EOT
    LiteLLM enterprise license string. When set, the stack creates a
    `<tenant>-litellm-<env>-license` Key Vault secret, grants the
    Container Apps managed identity read access, and exposes its value to
    gateway + backend as `LITELLM_LICENSE`. Leave empty for OSS-only deploys.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "ui_password" {
  description = <<-EOT
    UI admin password. When set, the stack creates a
    `<tenant>-litellm-<env>-ui-password` Key Vault secret, grants the
    Container Apps managed identity read access, and exposes its value to
    the backend as `UI_PASSWORD`. Pair with `backend_extra_env.UI_USERNAME`
    to set the matching username. Leave empty to skip; the proxy then
    falls back to the LITELLM_MASTER_KEY for UI login.
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

# ---------- Networking ----------

variable "vnet_cidr" {
  description = "CIDR block for the VNet."
  type        = string
  default     = "10.42.0.0/16"
}

variable "azs" {
  description = "Availability zones to spread subnets across. Use Azure zone identifiers (e.g. `[\"1\", \"2\", \"3\"]`). At least 1 required; 2+ recommended for HA PostgreSQL Flexible Server."
  type        = list(string)

  validation {
    condition     = length(var.azs) >= 1
    error_message = "Provide at least 1 availability zone."
  }
}

# ---------- Component images ----------
#
# Defaults pin the four componentized images at the same release tag on
# GHCR. Override on a per-component basis in tfvars when bumping; bump them
# together when bumping the LiteLLM release.
#
# Container Apps pulls `ghcr.io/berriai/litellm-gateway:<tag>` by default;
# this works as-is in Azure because Container Apps can pull from ghcr.io
# directly (the GCP stack requires an Artifact Registry mirror, AWS pulls
# from ECR only after a docker push, Azure is the closest to a direct
# pull-from-public-registry experience).

variable "image_registry" {
  description = "Container registry host (without trailing slash). Defaults to GHCR; override when mirroring to ACR or another private registry."
  type        = string
  default     = "ghcr.io"
}

variable "image_namespace" {
  description = "Container registry namespace / owner. Defaults to `berriai` on GHCR."
  type        = string
  default     = "berriai"
}

variable "image_tag" {
  description = "Image tag (e.g. `v1.86.0-dev`). All four component images use the same tag; bump together when bumping LiteLLM."
  type        = string
  default     = "latest"
}

variable "gateway_image" {
  description = "Container image for the gateway (data plane, port 4000). Defaults to `<image_registry>/<image_namespace>/litellm-gateway:<image_tag>`."
  type        = string
  default     = ""
}

variable "backend_image" {
  description = "Container image for the backend (management API, port 4001). Defaults to `<image_registry>/<image_namespace>/litellm-backend:<image_tag>`."
  type        = string
  default     = ""
}

variable "ui_image" {
  description = "Container image for the UI (port 3000). Defaults to `<image_registry>/<image_namespace>/litellm-ui:<image_tag>`."
  type        = string
  default     = ""
}

variable "migrations_image" {
  description = "Container image for the one-off migration job. Defaults to `<image_registry>/<image_namespace>/litellm-migrations:<image_tag>`."
  type        = string
  default     = ""
}

# ---------- Database ----------

variable "db_sku_name" {
  description = "Azure Database for PostgreSQL Flexible Server SKU (e.g. `Standard_B1ms`, `Standard_D2s_v3`, `GP_Standard_D2s_v3`)."
  type        = string
  default     = "Standard_B1ms"
}

variable "db_version" {
  description = "PostgreSQL major version (e.g. `15`, `16`, `17`)."
  type        = string
  default     = "16"
}

variable "db_storage_mb" {
  description = "Storage size in MB for the PostgreSQL Flexible Server."
  type        = number
  default     = 32768
}

variable "db_name" {
  description = "Database name to create on the PostgreSQL Flexible Server."
  type        = string
  default     = "litellm"
}

variable "db_username" {
  description = "Application DB username. The Container Apps managed identity uses Entra (Azure AD) token auth at runtime; this user is created during `terraform apply` by the bootstrap job and granted the rights the proxy needs."
  type        = string
  default     = "litellm_app"
}

# ---------- Redis ----------

variable "redis_sku" {
  description = "Azure Cache for Redis SKU. Defaults to `Basic` (single node, dev/trial); `Standard` or `Premium` recommended for production (zone redundancy + replication)."
  type        = string
  default     = "Basic"
}

variable "redis_family" {
  description = "Redis SKU family. `C` is basic; `P` is premium."
  type        = string
  default     = "C"
}

variable "redis_capacity" {
  description = "Cache capacity for Azure Cache for Redis (0 = 250MB, 1 = 1GB, ...). Match to your expected throughput."
  type        = number
  default     = 0
}

variable "redis_enable_ssl" {
  description = "Whether to require SSL for Redis connections. Defaults to true to match AWS module's `transit_encryption_enabled`."
  type        = bool
  default     = true
}

# ---------- Storage ----------

variable "storage_force_destroy" {
  description = "Allow `terraform destroy` to delete a non-empty Azure Storage container (and the Storage Account). Off by default to protect cached responses, archived request logs, and /v1/files storage."
  type        = bool
  default     = false
}

variable "storage_account_tier" {
  description = "Storage Account performance tier. `Standard` is fine for cache / log archive / file storage; `Premium` for low-latency workloads."
  type        = string
  default     = "Standard"
}

variable "storage_replication_type" {
  description = "Storage Account replication type (`LRS`, `GRS`, `RAGRS`, `ZRS`). Defaults to LRS for cost; bump to GRS / ZRS for production durability."
  type        = string
  default     = "LRS"
}

# ---------- TLS / Load balancer ----------

variable "key_vault_certificate_id" {
  description = <<-EOT
    Resource ID of an existing Key Vault certificate to attach to the
    Application Gateway listener for TLS termination. When unset, the
    Application Gateway listener uses HTTP only; either set this var OR
    set `allow_plaintext_app_gateway = true` for dev/trial only.
  EOT
  type        = string
  default     = ""
}

variable "allow_plaintext_app_gateway" {
  description = "Allow the Application Gateway listener to serve HTTP (port 80) without TLS. Dev / trial only. Defaults to false to match the AWS module's secure default."
  type        = bool
  default     = false
}

# ---------- Compute sizing ----------

variable "gateway_cpu" {
  description = "CPU allocation for the gateway Container App in 0.25 vCPU increments (e.g. `0.5`, `1.0`, `2.0`)."
  type        = number
  default     = 1.0
}

variable "gateway_memory" {
  description = "Memory for the gateway Container App, e.g. `2Gi`, `4Gi`."
  type        = string
  default     = "2Gi"
}

variable "gateway_min_replicas" {
  description = "Minimum replicas for the gateway Container App (autoscaler lower bound)."
  type        = number
  default     = 2
}

variable "gateway_max_replicas" {
  description = "Maximum replicas for the gateway Container App (autoscaler upper bound)."
  type        = number
  default     = 10
}

variable "backend_cpu" {
  description = "CPU allocation for the backend Container App."
  type        = number
  default     = 1.0
}

variable "backend_memory" {
  description = "Memory for the backend Container App."
  type        = string
  default     = "2Gi"
}

variable "backend_min_replicas" {
  description = "Minimum replicas for the backend Container App."
  type        = number
  default     = 1
}

variable "backend_max_replicas" {
  description = "Maximum replicas for the backend Container App."
  type        = number
  default     = 5
}

variable "ui_cpu" {
  description = "CPU allocation for the UI Container App."
  type        = number
  default     = 0.5
}

variable "ui_memory" {
  description = "Memory for the UI Container App."
  type        = string
  default     = "1Gi"
}

variable "ui_min_replicas" {
  description = "Minimum replicas for the UI Container App."
  type        = number
  default     = 1
}

variable "ui_max_replicas" {
  description = "Maximum replicas for the UI Container App."
  type        = number
  default     = 3
}

# ---------- proxy_config and extra config ----------

variable "proxy_config" {
  description = <<-EOT
    Mirrors the helm chart's `gateway.config.proxy_config`. The map is
    YAML-encoded and uploaded to the storage account blob `config/litellm-config.yaml`;
    the gateway and backend Container Apps download it to
    `/tmp/litellm-config.yaml` at startup via the SDK and set
    `CONFIG_FILE_PATH` to match. Editing this value produces a new
    Container Apps revision and rolling redeploy of both services.
  EOT
  type        = any
  default     = {}
}

variable "gateway_extra_env" {
  description = "Extra non-sensitive env vars (plaintext) merged into the gateway Container App env. Useful for feature flags / observability hosts. Provider API keys belong in gateway_extra_secrets."
  type        = map(string)
  default     = {}
}

variable "backend_extra_env" {
  description = "Extra non-sensitive env vars merged into the backend Container App env."
  type        = map(string)
  default     = {}
}

variable "gateway_extra_secrets" {
  description = "Map of Key Vault secret references to mount into the gateway Container App as env vars, e.g. `{ OPENAI_API_KEY = azurerm_key_vault_secret.openai.id }`. The Container Apps managed identity must have Secret User role on the vault."
  type        = map(string)
  default     = {}
}

variable "backend_extra_secrets" {
  description = "Map of Key Vault secret references to mount into the backend Container App."
  type        = map(string)
  default     = {}
}

variable "ui_extra_env" {
  description = "Extra non-sensitive env vars merged into the ui Container App env (frontend feature flags, etc.)."
  type        = map(string)
  default     = {}
}

# ---------- Observability ----------

variable "otel_endpoint" {
  description = "OTLP endpoint (e.g. `https://otel.example.com:4317`). When empty, OpenTelemetry instrumentation is disabled (matches the AWS module)."
  type        = string
  default     = ""
}

variable "otel_exporter" {
  description = "OTEL exporter protocol. `otlp` or `otlp_http`."
  type        = string
  default     = "otlp"
}

variable "otel_environment_name" {
  description = "OTEL environment tag (typically the deployment environment: `prod`, `stage`, ...). Defaults to var.env."
  type        = string
  default     = ""
}

variable "otel_capture_message_content" {
  description = "Whether to capture message content in spans (PII consideration)."
  type        = string
  default     = "false"
}

variable "log_retention_days" {
  description = "Log Analytics workspace retention in days (applies to application logs routed through the workspace)."
  type        = number
  default     = 30
}
