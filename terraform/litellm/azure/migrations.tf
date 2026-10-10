# Container Apps Job for the one-off prisma migrate deploy. The job runs
# once after `terraform apply` and before any traffic; output the run
# command via `terraform output migration_run_command` and execute it
# via `az containerapp job start`.
#
# Like the AWS migrations task definition, this container downloads the
# proxy_config.yaml blob via the managed identity at startup, runs the
# schema migration, then exits. The downstream gateway / backend services
# do not depend on its success at terraform-apply time because the
# Container Apps revision bootstrap happens lazily; instead we rely on
# the bootstrap pattern in the README (run after apply, before first
# traffic).

resource "azurerm_container_app_job" "migrations" {
  name                         = "${local.name}-migrations"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = local.resource_group_name
  location                     = var.location
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  # Manual trigger: one-shot job, runs to completion when invoked.
  manual_trigger_config {
    parallelism              = 1
    replica_completion_count = 1
  }

  replica_timeout_in_seconds = 1800

  registry {
    server = var.image_registry
  }

  # Secrets are declared at the job level and referenced from container
  # env blocks via `secret_name`. Key Vault secret IDs are pulled at
  # startup using the user-assigned managed identity declared on the job.
  dynamic "secret" {
    for_each = local.migrations_kv_secret_env
    content {
      name                = secret.key
      key_vault_secret_id = secret.value
      identity            = azurerm_user_assigned_identity.container_apps.id
    }
  }

  template {
    container {
      name   = "migrations"
      image  = local.migrations_image_resolved
      cpu    = 1.0
      memory = "2Gi"

      dynamic "env" {
        for_each = concat(
          local.migrations_base_env,
          [
            { name = "DATABASE_URL", value = "postgresql://${var.db_username}@${azurerm_postgresql_flexible_server.this.fqdn}:5432/${var.db_name}?sslmode=require" },
          ],
        )
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      dynamic "env" {
        for_each = local.migrations_kv_secret_env
        content {
          name        = env.key
          secret_name = env.key
        }
      }
    }
  }
}
