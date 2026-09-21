# Plan-only coverage for the in-container PgBouncer knobs on the gateway
# service. `mock_provider` keeps this offline: no GCP credentials, no API
# calls, no resources. Run from terraform/litellm/gcp with `terraform test`.

mock_provider "google" {
  mock_resource "google_redis_instance" {
    defaults = {
      host = "10.0.0.4"
      port = 6379
      server_ca_certs = [{
        cert = "-----BEGIN CERTIFICATE-----\nmock\n-----END CERTIFICATE-----"
      }]
    }
  }
}

mock_provider "google-beta" {}
mock_provider "random" {}

variables {
  project_id         = "test-project"
  tenant             = "tenant"
  env                = "test"
  allow_plaintext_lb = true
  image_registry     = "us-central1-docker.pkg.dev/test-project/litellm"
}

run "pool_off_by_default" {
  command = plan

  assert {
    condition     = length(local.gateway_pool_env) == 0
    error_message = "The gateway must get no LITELLM_PGBOUNCER_* env unless gateway_connection_pool_enabled is set."
  }

  assert {
    condition = !anytrue([
      for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : startswith(e.name, "LITELLM_PGBOUNCER_")
    ])
    error_message = "The gateway service must carry no LITELLM_PGBOUNCER_* env by default."
  }
}

run "pool_enabled_renders_the_three_vars_with_configured_sizes" {
  command = plan

  variables {
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

  assert {
    condition = alltrue([
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : e.name], "LITELLM_PGBOUNCER_ENABLED"),
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : e.name], "LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS"),
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : e.name], "LITELLM_PGBOUNCER_MAX_CLIENT_CONN"),
    ])
    error_message = "The gateway service must receive all three LITELLM_PGBOUNCER_* env vars."
  }

  assert {
    condition = !anytrue(concat(
      [for e in google_cloud_run_v2_service.backend[0].template[0].containers[0].env : startswith(e.name, "LITELLM_PGBOUNCER_")],
      [for e in google_cloud_run_v2_job.migrations[0].template[0].template[0].containers[0].env : startswith(e.name, "LITELLM_PGBOUNCER_")],
    ))
    error_message = "The backend service and the migrations job must keep their direct database connection."
  }
}

run "collector_sidecar_gets_the_same_pool_env_as_the_gateway" {
  command = plan

  variables {
    collector_enabled               = true
    gateway_connection_pool_enabled = true
    gateway_pool_max_db_connections = 8
    gateway_pool_max_client_conn    = 250
  }

  assert {
    condition = alltrue([
      for c in google_cloud_run_v2_service.gateway[0].template[0].containers : (
        { for e in c.env : e.name => e.value }["LITELLM_PGBOUNCER_ENABLED"] == "true" &&
        { for e in c.env : e.name => e.value }["LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS"] == "8" &&
        { for e in c.env : e.name => e.value }["LITELLM_PGBOUNCER_MAX_CLIENT_CONN"] == "250"
      ) if c.name == "spend-collector"
    ]) && length([for c in google_cloud_run_v2_service.gateway[0].template[0].containers : c.name if c.name == "spend-collector"]) == 1
    error_message = "The spend-collector sidecar must carry the same three LITELLM_PGBOUNCER_* vars as the gateway so its Prisma connects to the instance-local pool."
  }
}

run "collector_sidecar_gets_no_pool_env_when_the_pool_is_off" {
  command = plan

  variables {
    collector_enabled = true
  }

  assert {
    condition = !anytrue(flatten([
      for c in google_cloud_run_v2_service.gateway[0].template[0].containers : [
        for e in c.env : startswith(e.name, "LITELLM_PGBOUNCER_")
      ] if c.name == "spend-collector"
    ])) && length([for c in google_cloud_run_v2_service.gateway[0].template[0].containers : c.name if c.name == "spend-collector"]) == 1
    error_message = "The spend-collector sidecar must get no LITELLM_PGBOUNCER_* env unless gateway_connection_pool_enabled is set."
  }
}

run "pool_enabled_uses_the_module_default_sizes" {
  command = plan

  variables {
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
      endswith(google_cloud_run_v2_service.gateway[0].template[0].containers[0].args[0], local.gateway_launch_cmd),
    ])
    error_message = "The gateway must start through gateway.launch (with and without ddtrace) so the pooler starts once before uvicorn forks the workers."
  }

  assert {
    condition     = strcontains(local.backend_launch_cmd, "uvicorn backend.main:app")
    error_message = "The backend has no workers to share a pooler and keeps starting uvicorn directly."
  }
}

run "pool_sizes_below_one_fail_at_plan" {
  command = plan

  variables {
    gateway_connection_pool_enabled = true
    gateway_pool_max_db_connections = 0
    gateway_pool_max_client_conn    = 0
  }

  expect_failures = [
    var.gateway_pool_max_db_connections,
    var.gateway_pool_max_client_conn,
  ]
}
