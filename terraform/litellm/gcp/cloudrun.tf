# Three Cloud Run v2 services + one Cloud Run v2 job for migrations.
# All four use the same service account and the same VPC connector for
# private egress to Cloud SQL + Memorystore.

locals {
  # Memorystore exposes a self-signed CA cert per instance; we ship it as
  # a base64 env var and decode it to a file at container startup so the
  # rediss:// connection can validate. Public cert, not sensitive.
  redis_ca_pem_b64 = base64encode(google_redis_instance.this.server_ca_certs[0].cert)

  shared_env_kv = [
    { name = "DATABASE_HOST", value = google_sql_database_instance.writer.private_ip_address },
    { name = "DATABASE_PORT", value = "5432" },
    { name = "DATABASE_USER", value = var.db_username },
    { name = "DATABASE_NAME", value = var.db_name },
    { name = "DATABASE_HOST_READ_REPLICA", value = google_sql_database_instance.reader.private_ip_address },
    { name = "DATABASE_PORT_READ_REPLICA", value = "5432" },
    { name = "REDIS_HOST", value = google_redis_instance.this.host },
    { name = "REDIS_PORT", value = tostring(google_redis_instance.this.port) },
    # _redis.get_redis_url_from_environment honors REDIS_SSL to flip the
    # scheme to rediss://; REDIS_SSL_CA_CERTS is mapped via
    # _get_redis_env_kwarg_mapping → ssl_ca_certs on the redis-py client.
    { name = "REDIS_SSL", value = "true" },
    { name = "REDIS_SSL_CA_CERTS", value = "/tmp/redis-ca.pem" },
    { name = "REDIS_CA_PEM_B64", value = local.redis_ca_pem_b64 },
    { name = "GCS_BUCKET_NAME", value = google_storage_bucket.this.name },
  ]

  # Cloud Run v2 secret env vars use value_source.secret_key_ref pointing at a
  # secret resource ID. Shared between gateway and backend (the migrations
  # job has its own narrower env list — see migrations_env_secrets below).
  shared_env_secrets = concat(
    [
      { name = "LITELLM_MASTER_KEY", secret = google_secret_manager_secret.master_key.id, version = "latest" },
      { name = "DATABASE_PASSWORD", secret = google_secret_manager_secret.db_password.id, version = "latest" },
    ],
    var.litellm_license == "" ? [] : [
      { name = "LITELLM_LICENSE", secret = google_secret_manager_secret.license[0].id, version = "latest" },
    ],
  )

  # Backend-only managed secrets. UI_PASSWORD is consumed by the management
  # API (UI login flow) and has no use on the gateway data plane.
  backend_managed_env_secrets = var.ui_password == "" ? [] : [
    { name = "UI_PASSWORD", secret = google_secret_manager_secret.ui_password[0].id, version = "latest" },
  ]

  # Per-component extras (from variables).
  gateway_extra_env_kv = [
    for k, v in var.gateway_extra_env : { name = k, value = v }
  ]
  backend_extra_env_kv = [
    for k, v in var.backend_extra_env : { name = k, value = v }
  ]

  backend_default_env_kv = [
    { name = "STORE_MODEL_IN_DB", value = "true" },
  ]
  gateway_extra_secret_kv = [
    for k, v in var.gateway_extra_secrets : { name = k, secret = v, version = "latest" }
  ]
  backend_extra_secret_kv = [
    for k, v in var.backend_extra_secrets : { name = k, secret = v, version = "latest" }
  ]

  # Shell fragments composed with && so any failure short-circuits the
  # whole startup instead of falling through to `exec uvicorn`. The
  # python step is only included when the caller provided a proxy_config.
  proxy_config_fragment = local.proxy_config_enabled ? [
    "python -c \"import os, base64, pathlib; pathlib.Path(os.environ['CONFIG_FILE_PATH']).write_bytes(base64.b64decode(os.environ['LITELLM_PROXY_CONFIG_B64']))\""
  ] : []

  # Decode the Memorystore CA cert (passed as REDIS_CA_PEM_B64) to the
  # path REDIS_SSL_CA_CERTS points at, so the redis-py client can validate
  # the rediss:// handshake.
  redis_ca_fragment = [
    "python -c \"import os, base64, pathlib; pathlib.Path(os.environ['REDIS_SSL_CA_CERTS']).write_bytes(base64.b64decode(os.environ['REDIS_CA_PEM_B64']))\""
  ]

  database_url_fragment = [
    "export DATABASE_URL=\"postgresql://$${DATABASE_USER}:$${DATABASE_PASSWORD}@$${DATABASE_HOST}:$${DATABASE_PORT}/$${DATABASE_NAME}\"",
    "export DATABASE_URL_READ_REPLICA=\"postgresql://$${DATABASE_USER}:$${DATABASE_PASSWORD}@$${DATABASE_HOST_READ_REPLICA}:$${DATABASE_PORT_READ_REPLICA}/$${DATABASE_NAME}\"",
  ]

  gateway_args = join(" && ", concat(
    local.proxy_config_fragment,
    local.redis_ca_fragment,
    local.database_url_fragment,
    ["exec uvicorn gateway.main:app --host 0.0.0.0 --port 4000"],
  ))

  backend_args = join(" && ", concat(
    local.proxy_config_fragment,
    local.redis_ca_fragment,
    local.database_url_fragment,
    ["exec uvicorn backend.main:app --host 0.0.0.0 --port 4001"],
  ))

  # Env shipped to the migrations Job. The migrations image runs run.py
  # which assembles DATABASE_URL from these discrete vars itself, so we
  # only need writer-side DB env (no read replica, no proxy_config, no
  # master key).
  migrations_env_kv = [
    { name = "DATABASE_HOST", value = google_sql_database_instance.writer.private_ip_address },
    { name = "DATABASE_PORT", value = "5432" },
    { name = "DATABASE_USER", value = var.db_username },
    { name = "DATABASE_NAME", value = var.db_name },
  ]

  migrations_env_secrets = [
    { name = "DATABASE_PASSWORD", secret = google_secret_manager_secret.db_password.id, version = "latest" },
  ]
}

# ---------- Gateway ----------
resource "google_cloud_run_v2_service" "gateway" {
  name     = "${local.name}-gateway"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account                  = google_service_account.runtime.email
    max_instance_request_concurrency = var.gateway_max_instance_request_concurrency

    vpc_access {
      connector = google_vpc_access_connector.this.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    scaling {
      min_instance_count = var.gateway_min_instances
      max_instance_count = var.gateway_max_instances
    }

    containers {
      image   = local.gateway_image
      command = ["sh", "-c"]
      args    = [local.gateway_args]

      ports {
        container_port = 4000
      }

      resources {
        limits = {
          cpu    = var.gateway_cpu
          memory = var.gateway_memory
        }
      }

      dynamic "env" {
        for_each = concat(local.shared_env_kv, local.gateway_extra_env_kv, local.proxy_config_env)
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      dynamic "env" {
        for_each = concat(local.shared_env_secrets, local.gateway_extra_secret_kv)
        content {
          name = env.value.name
          value_source {
            secret_key_ref {
              secret  = env.value.secret
              version = env.value.version
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health/readiness"
          port = 4000
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 12
      }

      liveness_probe {
        http_get {
          path = "/health/liveliness"
          port = 4000
        }
        period_seconds  = 30
        timeout_seconds = 5
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.master_key,
    google_secret_manager_secret_iam_member.db_password,
    google_secret_manager_secret_iam_member.license,
    google_secret_manager_secret_iam_member.extras,
    google_sql_user.app,
    # Don't go live until the schema is migrated; otherwise the proxy boots,
    # fails on missing tables, and Cloud Run keeps cold-restarting.
    terraform_data.migration,
  ]
}

# ---------- Backend ----------
resource "google_cloud_run_v2_service" "backend" {
  name     = "${local.name}-backend"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account                  = google_service_account.runtime.email
    max_instance_request_concurrency = var.backend_max_instance_request_concurrency

    vpc_access {
      connector = google_vpc_access_connector.this.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    scaling {
      min_instance_count = var.backend_min_instances
      max_instance_count = var.backend_max_instances
    }

    containers {
      image   = local.backend_image
      command = ["sh", "-c"]
      args    = [local.backend_args]

      ports {
        container_port = 4001
      }

      resources {
        limits = {
          cpu    = var.backend_cpu
          memory = var.backend_memory
        }
      }

      dynamic "env" {
        for_each = concat(local.shared_env_kv, local.backend_default_env_kv, local.backend_extra_env_kv, local.proxy_config_env)
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      dynamic "env" {
        for_each = concat(local.shared_env_secrets, local.backend_managed_env_secrets, local.backend_extra_secret_kv)
        content {
          name = env.value.name
          value_source {
            secret_key_ref {
              secret  = env.value.secret
              version = env.value.version
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health/readiness"
          port = 4001
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 12
      }

      liveness_probe {
        http_get {
          path = "/health/liveliness"
          port = 4001
        }
        period_seconds  = 30
        timeout_seconds = 5
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.master_key,
    google_secret_manager_secret_iam_member.db_password,
    google_secret_manager_secret_iam_member.license,
    google_secret_manager_secret_iam_member.ui_password,
    google_secret_manager_secret_iam_member.extras,
    google_sql_user.app,
    terraform_data.migration,
  ]
}

# ---------- UI ----------
# Static nginx — no DB, no Redis, no secrets. Runs as ui_runtime, a SA
# with zero IAM bindings, so a compromised UI container can't pivot to
# Secret Manager / Cloud SQL via the metadata service.
resource "google_cloud_run_v2_service" "ui" {
  name     = "${local.name}-ui"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account                  = google_service_account.ui_runtime.email
    max_instance_request_concurrency = var.ui_max_instance_request_concurrency

    scaling {
      min_instance_count = var.ui_min_instances
      max_instance_count = var.ui_max_instances
    }

    containers {
      image = local.ui_image

      ports {
        container_port = 3000
      }

      resources {
        limits = {
          cpu    = var.ui_cpu
          memory = var.ui_memory
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
          port = 3000
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        timeout_seconds       = 3
        failure_threshold     = 6
      }
    }
  }
}

# Allow the LB (any unauthenticated traffic from the configured serverless
# NEG) to invoke the Cloud Run services. The actual auth is in the proxy
# (LITELLM_MASTER_KEY); these IAM bindings just open up Cloud Run's invoker
# gate so the LB request makes it to the container.
resource "google_cloud_run_v2_service_iam_member" "gateway_allusers" {
  project  = var.project
  location = google_cloud_run_v2_service.gateway.location
  name     = google_cloud_run_v2_service.gateway.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "backend_allusers" {
  project  = var.project
  location = google_cloud_run_v2_service.backend.location
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "ui_allusers" {
  project  = var.project
  location = google_cloud_run_v2_service.ui.location
  name     = google_cloud_run_v2_service.ui.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------- Migrations job ----------
# Dedicated litellm-migrations image — slim, ENTRYPOINT runs run.py which
# assembles DATABASE_URL from the DATABASE_* env vars and runs `prisma
# migrate deploy`. No proxy_config, no master key, no shell wrapper.
resource "google_cloud_run_v2_job" "migrations" {
  name     = "${local.name}-migrations"
  location = var.region

  template {
    template {
      service_account = google_service_account.runtime.email

      vpc_access {
        connector = google_vpc_access_connector.this.id
        egress    = "PRIVATE_RANGES_ONLY"
      }

      containers {
        image = local.migrations_image

        # Prisma's Node + Rust engine plus the v2 migration resolver
        # routinely peaks above 1 GiB while applying the schema, so 2 GiB
        # is the floor — 1 GiB OOM-kills mid-migrate. CPU stays at 1 vCPU
        # (Cloud Run requires >= 1 with concurrency > 1, and `prisma
        # migrate deploy` is single-threaded so more buys nothing).
        resources {
          limits = {
            cpu    = "1000m"
            memory = "4Gi"
          }
        }

        dynamic "env" {
          for_each = local.migrations_env_kv
          content {
            name  = env.value.name
            value = env.value.value
          }
        }

        dynamic "env" {
          for_each = local.migrations_env_secrets
          content {
            name = env.value.name
            value_source {
              secret_key_ref {
                secret  = env.value.secret
                version = env.value.version
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.db_password,
    google_sql_user.app,
  ]
}
