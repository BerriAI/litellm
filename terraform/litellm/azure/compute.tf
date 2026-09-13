# Container Apps Environment (one per stack) + the three componentized
# apps (gateway / backend / ui) plus the migrations job.
#
# The Azure equivalent of AWS ECS Fargate is Azure Container Apps: a
# serverless managed-Kubernetes-ish runtime that runs containers, has
# built-in HTTPS ingress with cert-managed-by-Azure, autoscaling, and
# native Azure AD / managed-identity support. The ACA infrastructure
# subnet must be /23 or larger; we already carved a /20 in network.tf.

# ---------- Shared environment ----------
#
# Observability: when var.otel_endpoint is set, OTel instrumentation
# turns on for both gateway and backend. Service names match the AWS
# stack so spans land in the right bucket.

locals {
  otel_enabled          = var.otel_endpoint != ""
  otel_environment_name = var.otel_environment_name != "" ? var.otel_environment_name : var.env
  otel_shared_env_raw = local.otel_enabled ? [
    { name = "LITELLM_OTEL_V2", value = "true" },
    { name = "OTEL_EXPORTER", value = var.otel_exporter },
    { name = "OTEL_ENDPOINT", value = var.otel_endpoint },
    { name = "OTEL_ENVIRONMENT_NAME", value = local.otel_environment_name },
    { name = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", value = var.otel_capture_message_content },
  ] : []
  gateway_otel_env = [
    for e in local.otel_shared_env_raw : e if !contains(keys(var.gateway_extra_env), e.name)
  ]
  backend_otel_env = [
    for e in local.otel_shared_env_raw : e if !contains(keys(var.backend_extra_env), e.name)
  ]

  # Shared Postgres + Redis + Storage env, fed to gateway / backend /
  # migrations. Matches the AWS module's `shared_env` shape for those
  # values that map to Azure equivalents.
  shared_pg_env = [
    { name = "DATABASE_HOST", value = azurerm_postgresql_flexible_server.this.fqdn },
    { name = "DATABASE_PORT", value = "5432" },
    { name = "DATABASE_USER", value = var.db_username },
    { name = "DATABASE_NAME", value = var.db_name },
    # Azure uses Entra-managed-identity token auth at runtime
    # (azure-identity's DefaultAzureCredential). The proxy's
    # init_azure_db_url_from_env equivalent mints a short-lived token and
    # assembles DATABASE_URL when DATABASE_AZURE_AUTH=true is set.
    { name = "DATABASE_AZURE_AUTH", value = "true" },
  ]
  shared_redis_env = [
    { name = "REDIS_HOST", value = azurerm_redis_cache.this.hostname },
    { name = "REDIS_PORT", value = tostring(azurerm_redis_cache.this.ssl_port) },
    { name = "REDIS_SSL", value = "true" },
  ]
  shared_storage_env = [
    # LiteLLM reads AZURE_STORAGE_ACCOUNT_NAME and the AZURE_BLOB_*
    # container env vars. Auth is via the Container Apps managed identity
    # (Storage Blob Data Contributor role, see iam.tf).
    { name = "AZURE_STORAGE_ACCOUNT_NAME", value = azurerm_storage_account.this.name },
    { name = "AZURE_BLOB_STORAGE_ACCOUNT_NAME", value = azurerm_storage_account.this.name },
    { name = "AZURE_BLOB_STORAGE_CONTAINER_NAME", value = azurerm_storage_container.proxy.name },
    { name = "AZURE_FILE_STORAGE_CONTAINER_NAME", value = azurerm_storage_container.files.name },
    # CONFIG_FILE_PATH is read by the LiteLLM entrypoint to know where
    # to download the uploaded proxy_config.yaml blob to.
    { name = "CONFIG_FILE_PATH", value = "/tmp/litellm-config.yaml" },
  ]

  # Combine env plus extra_env (caller override) for each component.
  gateway_base_env = concat(
    local.shared_pg_env,
    local.shared_redis_env,
    local.shared_storage_env,
    local.gateway_otel_env,
  )
  backend_base_env = concat(
    local.shared_pg_env,
    local.shared_redis_env,
    local.shared_storage_env,
    local.backend_otel_env,
  )
  migrations_base_env = concat(
    local.shared_pg_env,
    local.shared_redis_env,
  )

  # Reference secret IDs (Key Vault secret URIs) for env vars that come
  # from Key Vault instead of plaintext. Each entry becomes a valueFrom
  # in the Container App env block.
  gateway_kv_secret_env = merge(
    {
      LITELLM_MASTER_KEY = azurerm_key_vault_secret.master_key.id
    },
    var.litellm_license != "" ? { LITELLM_LICENSE = one(azurerm_key_vault_secret.license[*].id) } : {},
    var.gateway_extra_secrets,
  )
  backend_kv_secret_env = merge(
    {
      LITELLM_MASTER_KEY = azurerm_key_vault_secret.master_key.id,
    },
    var.ui_password != "" ? { UI_PASSWORD = one(azurerm_key_vault_secret.ui_password[*].id) } : {},
    var.litellm_license != "" ? { LITELLM_LICENSE = one(azurerm_key_vault_secret.license[*].id) } : {},
    var.backend_extra_secrets,
  )
  migrations_kv_secret_env = {
    LITELLM_MASTER_KEY = azurerm_key_vault_secret.master_key.id
  }
}

# ---------- Container Apps Environment ----------

resource "azurerm_container_app_environment" "this" {
  name                = "${local.name}-cae"
  location            = var.location
  resource_group_name = local.resource_group_name
  tags                = local.tags

  infrastructure_subnet_id = azurerm_subnet.containers.id

  depends_on = [azurerm_role_assignment.container_apps_env_infra]
}

# ---------- Log Analytics workspace ----------
#
# Container Apps stdout/stderr route to Log Analytics. The default
# workspace created here is per-stack; callers that want to centralize
# logs can pass `log_analytics_workspace_id` (see variables) but that's
# out of scope for the baseline.

resource "azurerm_log_analytics_workspace" "this" {
  name                = "${local.name}-logs"
  location            = var.location
  resource_group_name = local.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days
  tags                = local.tags
}

# ---------- gateway ----------

resource "azurerm_container_app" "gateway" {
  name                         = "${local.name}-gateway"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = local.resource_group_name
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  # The gateway is internal-only; the Application Gateway front ends it
  # and fronts the UI; the backend is reachable from gateway + ui only.
  ingress {
    allow_insecure_connections = false
    external_enabled           = false
    target_port                = 4000
    transport                  = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  registry {
    server = var.image_registry
  }

  template {
    min_replicas = var.gateway_min_replicas
    max_replicas = var.gateway_max_replicas

    container {
      name   = "gateway"
      image  = local.gateway_image_resolved
      cpu    = var.gateway_cpu
      memory = var.gateway_memory

      dynamic "env" {
        for_each = concat(
          local.gateway_base_env,
          [for k, v in var.gateway_extra_env : { name = k, value = v }],
        )
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      # Key Vault references resolved at startup using the managed identity.
      dynamic "env" {
        for_each = { for k, v in local.gateway_kv_secret_env : k => v }
        content {
          name         = env.key
          secret_value = env.value
        }
      }
    }
  }

  depends_on = [
    azurerm_key_vault_access_policy.container_apps,
  ]
}

# ---------- backend ----------

resource "azurerm_container_app" "backend" {
  name                         = "${local.name}-backend"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = local.resource_group_name
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  ingress {
    allow_insecure_connections = false
    external_enabled           = false
    target_port                = 4001
    transport                  = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  registry {
    server = var.image_registry
  }

  template {
    min_replicas = var.backend_min_replicas
    max_replicas = var.backend_max_replicas

    container {
      name   = "backend"
      image  = local.backend_image_resolved
      cpu    = var.backend_cpu
      memory = var.backend_memory

      dynamic "env" {
        for_each = concat(
          local.backend_base_env,
          [for k, v in var.backend_extra_env : { name = k, value = v }],
        )
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      dynamic "env" {
        for_each = { for k, v in local.backend_kv_secret_env : k => v }
        content {
          name         = env.key
          secret_value = env.value
        }
      }
    }
  }
}

# ---------- ui ----------

resource "azurerm_container_app" "ui" {
  name                         = "${local.name}-ui"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = local.resource_group_name
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  ingress {
    allow_insecure_connections = false
    external_enabled           = false
    target_port                = 3000
    transport                  = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  registry {
    server = var.image_registry
  }

  template {
    min_replicas = var.ui_min_replicas
    max_replicas = var.ui_max_replicas

    container {
      name   = "ui"
      image  = local.ui_image_resolved
      cpu    = var.ui_cpu
      memory = var.ui_memory

      dynamic "env" {
        for_each = var.ui_extra_env
        content {
          name  = env.key
          value = env.value
        }
      }
    }
  }
}
