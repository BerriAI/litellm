# Plan-only coverage for the in-container PgBouncer knobs on the gateway task.
# `mock_provider` keeps this offline: no AWS credentials, no API calls, no
# resources. Run from terraform/litellm/aws with `terraform test`.

mock_provider "aws" {
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }
}
mock_provider "random" {}

variables {
  region              = "us-east-1"
  tenant              = "acme"
  env                 = "test"
  azs                 = ["us-east-1a", "us-east-1b"]
  allow_plaintext_alb = true
}

run "pool_off_by_default" {
  command = plan

  assert {
    condition     = length(local.gateway_pool_env) == 0
    error_message = "The gateway must get no LITELLM_PGBOUNCER_* env unless gateway_connection_pool_enabled is set."
  }
}

run "pool_enabled_renders_the_three_vars_with_configured_sizes" {
  command = plan

  variables {
    create_database                 = false
    database_url                    = "postgresql://litellm:pw@db.internal:5432/litellm"
    gateway_num_workers             = 4
    gateway_connection_pool_enabled = true
    gateway_pool_max_db_connections = 8
    gateway_pool_max_client_conn    = 250
  }

  assert {
    condition = alltrue([
      length(local.gateway_pool_env) == 3,
      local.gateway_pool_env[0].name == "LITELLM_PGBOUNCER_ENABLED" && local.gateway_pool_env[0].value == "true",
      local.gateway_pool_env[1].name == "LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS" && local.gateway_pool_env[1].value == "8",
      local.gateway_pool_env[2].name == "LITELLM_PGBOUNCER_MAX_CLIENT_CONN" && local.gateway_pool_env[2].value == "250",
    ])
    error_message = "The pool env must carry the enabled flag and the configured sizes as strings."
  }
}

run "collector_sidecar_gets_the_same_pool_env_as_the_gateway" {
  command = plan

  variables {
    create_database                 = false
    database_url                    = "postgresql://litellm:pw@db.internal:5432/litellm"
    collector_enabled               = true
    gateway_connection_pool_enabled = true
    gateway_pool_max_db_connections = 8
    gateway_pool_max_client_conn    = 250
  }

  assert {
    condition = alltrue([
      for env in [local.gateway_environment, local.collector_container[0].environment] : (
        { for e in env : e.name => e.value }["LITELLM_PGBOUNCER_ENABLED"] == "true" &&
        { for e in env : e.name => e.value }["LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS"] == "8" &&
        { for e in env : e.name => e.value }["LITELLM_PGBOUNCER_MAX_CLIENT_CONN"] == "250"
      )
    ])
    error_message = "The collector sidecar must carry the same three LITELLM_PGBOUNCER_* vars as the gateway so its Prisma connects to the task-local pool."
  }
}

run "collector_sidecar_gets_no_pool_env_when_the_pool_is_off" {
  command = plan

  variables {
    collector_enabled = true
  }

  assert {
    condition     = !anytrue([for e in local.collector_container[0].environment : startswith(e.name, "LITELLM_PGBOUNCER_")])
    error_message = "The collector sidecar must get no LITELLM_PGBOUNCER_* env unless gateway_connection_pool_enabled is set."
  }
}

run "pool_enabled_uses_the_module_default_sizes" {
  command = plan

  variables {
    create_database                 = false
    database_url                    = "postgresql://litellm:pw@db.internal:5432/litellm"
    gateway_connection_pool_enabled = true
  }

  assert {
    condition = alltrue([
      length(local.gateway_pool_env) == 3,
      local.gateway_pool_env[1].name == "LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS" && local.gateway_pool_env[1].value == "20",
      local.gateway_pool_env[2].name == "LITELLM_PGBOUNCER_MAX_CLIENT_CONN" && local.gateway_pool_env[2].value == "1000",
    ])
    error_message = "The pool env must fall back to the module defaults of 20 upstream and 1000 client connections."
  }
}

run "gateway_starts_through_the_pool_aware_launcher" {
  command = plan

  variables {
    gateway_num_workers = 4
  }

  assert {
    condition = alltrue([
      strcontains(local.gateway_launch_cmd, "exec python -m gateway.launch --host 0.0.0.0 --port 4000 --workers 4"),
      strcontains(local.gateway_launch_cmd, "exec ddtrace-run python -m gateway.launch --host 0.0.0.0 --port 4000 --workers 4"),
      !strcontains(local.gateway_launch_cmd, "uvicorn gateway.main:app"),
      local.gateway_proxy_overrides.command[0] == local.gateway_launch_cmd,
    ])
    error_message = "The gateway must start through gateway.launch (with and without ddtrace) so the pooler starts once before uvicorn forks the workers."
  }
}

run "pool_with_module_created_iam_aurora_plans_with_both_the_pool_and_iam_auth" {
  command = plan

  variables {
    gateway_connection_pool_enabled = true
  }

  assert {
    condition = alltrue([
      length(local.gateway_pool_env) == 3,
      contains(local.managed_db_env, { name = "IAM_TOKEN_DB_AUTH", value = "true" }),
    ])
    error_message = "With the module-created Aurora the gateway must get the pool env alongside IAM token auth."
  }
}

run "pool_without_any_database_fails_at_plan" {
  command = plan

  variables {
    create_database                 = false
    gateway_connection_pool_enabled = true
  }

  expect_failures = [
    aws_ecs_task_definition.gateway,
  ]
}

run "module_created_iam_aurora_without_the_pool_still_plans" {
  command = plan

  assert {
    condition     = contains(local.managed_db_env, { name = "IAM_TOKEN_DB_AUTH", value = "true" })
    error_message = "Without the pool the module-created Aurora must keep IAM token auth."
  }
}
