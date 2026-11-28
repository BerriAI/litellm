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
  image_tag          = "v9.9.9"
  project_id         = "test-project"
  tenant             = "tenant"
  env                = "test"
  allow_plaintext_lb = true
  image_registry     = "us-central1-docker.pkg.dev/test-project/litellm"
}

run "default_creates_everything" {
  command = plan

  assert {
    condition = alltrue([
      length(google_compute_network.this) == 1,
      length(google_compute_subnetwork.this) == 1,
      length(google_compute_global_address.psa) == 1,
      length(google_service_networking_connection.psa) == 1,
      length(google_vpc_access_connector.this) == 1,
      length(google_cloud_run_v2_service.gateway) == 1,
      length(google_cloud_run_v2_service.backend) == 1,
      length(google_cloud_run_v2_service.ui) == 1,
      length(google_cloud_run_v2_job.migrations) == 1,
      length(google_compute_global_address.lb) == 1,
      length(terraform_data.migration) == 1,
      length(random_password.master_key) == 1,
      length(random_password.db_password) == 1,
      length(google_secret_manager_secret_version.master_key) == 1,
      length(google_secret_manager_secret_version.db_password) == 1,
      length(google_sql_user.app) == 1,
    ])
    error_message = "The default mode must create networking, runtime services, the load balancer, and migrations."
  }

  assert {
    condition = (
      google_service_account.runtime[0].account_id == "tenant-litellm-test-runtime" &&
      google_service_account.ui_runtime[0].account_id == "tenant-litellm-test-ui-runtime" &&
      google_project_iam_member.runtime_cloudsql[0].role == "roles/cloudsql.client"
    )
    error_message = "The default mode must create separate runtime and UI identities and grant the runtime identity Cloud SQL access."
  }

  assert {
    condition     = google_redis_instance.this.transit_encryption_mode == "SERVER_AUTHENTICATION"
    error_message = "Redis transit encryption must remain enabled by default."
  }

  assert {
    condition     = google_compute_global_forwarding_rule.http[0].load_balancing_scheme == "EXTERNAL_MANAGED" && google_cloud_run_v2_service_iam_member.gateway_allusers[0].member == "allUsers"
    error_message = "The default public frontend and Cloud Run invoker grant must continue to work."
  }

  assert {
    condition     = length(local.shared_env_kv) == 12
    error_message = "The default runtime environment must include GCS and the three Redis TLS entries."
  }
}

run "existing_service_accounts_require_no_project_iam_writes" {
  command = plan

  variables {
    runtime_service_account_email = "customer-runtime@test-project.iam.gserviceaccount.com"
    ui_service_account_email      = "customer-ui@test-project.iam.gserviceaccount.com"
    gateway_metrics_port          = 4001
  }

  assert {
    condition = alltrue([
      length(google_service_account.runtime) == 0,
      length(google_service_account.ui_runtime) == 0,
      length(google_project_iam_member.runtime_cloudsql) == 0,
      length(google_project_iam_member.runtime_metric_writer) == 0,
      length(google_project_iam_member.runtime_log_writer) == 0,
    ])
    error_message = "Using both existing service accounts must skip service account creation and all project IAM grants, even with metrics enabled."
  }

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.gateway[0].template[0].service_account == var.runtime_service_account_email,
      google_cloud_run_v2_service.backend[0].template[0].service_account == var.runtime_service_account_email,
      google_cloud_run_v2_job.migrations[0].template[0].template[0].service_account == var.runtime_service_account_email,
      google_cloud_run_v2_service.ui[0].template[0].service_account == var.ui_service_account_email,
      output.runtime_service_account_email == var.runtime_service_account_email,
      google_secret_manager_secret_iam_member.master_key.member == "serviceAccount:${var.runtime_service_account_email}",
      google_storage_bucket_iam_member.runtime.member == "serviceAccount:${var.runtime_service_account_email}",
    ])
    error_message = "Cloud Run, the migration job, and resource-level access must use the supplied service accounts."
  }
}

run "existing_ui_account_cannot_reuse_managed_runtime_identity" {
  command = plan

  variables {
    ui_service_account_email = "tenant-litellm-test-runtime@test-project.iam.gserviceaccount.com"
  }

  expect_failures = [google_cloud_run_v2_service.ui]
}

run "existing_runtime_account_cannot_reuse_managed_ui_identity" {
  command = plan

  variables {
    runtime_service_account_email = "tenant-litellm-test-ui-runtime@test-project.iam.gserviceaccount.com"
  }

  expect_failures = [google_cloud_run_v2_service.ui]
}

run "existing_service_account_still_requires_billing_credentials" {
  command = plan

  variables {
    runtime_service_account_email = "customer-runtime@test-project.iam.gserviceaccount.com"
    billing_metrics_endpoint      = "https://metrics.example.com"
  }

  expect_failures = [var.billing_metrics_endpoint]
}

run "deps_only_creates_no_runtime" {
  command = plan

  variables {
    create_runtime  = false
    private_ingress = {}
    proxy_config = {
      model_list = []
    }
  }

  assert {
    condition = alltrue([
      length(google_cloud_run_v2_service.gateway) == 0,
      length(google_cloud_run_v2_service.backend) == 0,
      length(google_cloud_run_v2_service.ui) == 0,
      length(google_cloud_run_v2_job.migrations) == 0,
      length(google_cloud_run_v2_service_iam_member.gateway_allusers) == 0,
      length(google_cloud_run_v2_service_iam_member.backend_allusers) == 0,
      length(google_cloud_run_v2_service_iam_member.ui_allusers) == 0,
      length(google_compute_global_address.lb) == 0,
      length(google_compute_region_network_endpoint_group.gateway) == 0,
      length(google_compute_region_network_endpoint_group.backend) == 0,
      length(google_compute_region_network_endpoint_group.ui) == 0,
      length(google_compute_backend_service.gateway) == 0,
      length(google_compute_backend_service.backend) == 0,
      length(google_compute_backend_service.ui) == 0,
      length(google_compute_url_map.this) == 0,
      length(google_compute_url_map.https_redirect) == 0,
      length(google_compute_target_http_proxy.this) == 0,
      length(google_compute_global_forwarding_rule.http) == 0,
      length(google_compute_managed_ssl_certificate.this) == 0,
      length(google_compute_target_https_proxy.this) == 0,
      length(google_compute_global_forwarding_rule.https) == 0,
      length(google_compute_forwarding_rule.private_lb) == 0,
      length(google_compute_subnetwork.lb_proxy) == 0,
      length(terraform_data.migration) == 0,
      length(google_vpc_access_connector.this) == 0,
      length(google_service_account.ui_runtime) == 0,
      length(google_storage_bucket.proxy_config) == 0,
    ])
    error_message = "Dependencies-only mode must omit all runtime, load balancer, connector, UI identity, and proxy config resources."
  }

  assert {
    condition = alltrue([
      google_sql_database_instance.writer.name == "tenant-litellm-test",
      google_sql_database_instance.reader.name == "tenant-litellm-test-reader",
      google_redis_instance.this.name == "tenant-litellm-test",
      google_storage_bucket.this.force_destroy == false,
      google_secret_manager_secret.master_key[0].secret_id == "tenant-litellm-test-master-key",
      google_secret_manager_secret.db_password[0].secret_id == "tenant-litellm-test-db-password",
      google_service_account.runtime[0].account_id == "tenant-litellm-test-runtime",
    ])
    error_message = "Dependencies-only mode must retain data stores, secrets, and the runtime service account."
  }

  assert {
    condition     = output.lb_url == null && output.migration_run_command == null
    error_message = "Runtime outputs must be null while dependency outputs remain available."
  }
}

run "existing_secrets_use_no_terraform_managed_passwords" {
  command = plan

  variables {
    litellm_master_key_secret_id = "projects/test-project/secrets/customer-master-key"
    db_password_secret_id        = "projects/test-project/secrets/customer-db-password"
  }

  assert {
    condition = alltrue([
      length(random_password.master_key) == 0,
      length(random_password.db_password) == 0,
      length(google_secret_manager_secret.master_key) == 0,
      length(google_secret_manager_secret.db_password) == 0,
      length(google_secret_manager_secret_version.master_key) == 0,
      length(google_secret_manager_secret_version.db_password) == 0,
      length(google_sql_user.app) == 0,
    ])
    error_message = "Customer-managed credentials must not create password values, secret versions, or a competing Cloud SQL user."
  }

  assert {
    condition = alltrue([
      google_secret_manager_secret_iam_member.master_key.secret_id == var.litellm_master_key_secret_id,
      google_secret_manager_secret_iam_member.db_password.secret_id == var.db_password_secret_id,
      output.master_key_secret_id == "customer-master-key",
      output.db_password_secret_id == "customer-db-password",
      one([for env in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : env.value_source[0].secret_key_ref[0].secret if env.name == "LITELLM_MASTER_KEY"]) == var.litellm_master_key_secret_id,
      one([for env in google_cloud_run_v2_service.backend[0].template[0].containers[0].env : env.value_source[0].secret_key_ref[0].secret if env.name == "DATABASE_PASSWORD"]) == var.db_password_secret_id,
      one([for env in google_cloud_run_v2_job.migrations[0].template[0].template[0].containers[0].env : env.value_source[0].secret_key_ref[0].secret if env.name == "DATABASE_PASSWORD"]) == var.db_password_secret_id,
    ])
    error_message = "Cloud Run services, migrations, and runtime secret access must use the supplied secret IDs."
  }
}

run "deps_only_needs_no_image_tag" {
  command = plan

  variables {
    create_runtime = false
    image_tag      = ""
  }

  assert {
    condition     = length(google_cloud_run_v2_service.gateway) == 0
    error_message = "deps-only mode must plan without an image tag since it deploys no images"
  }
}

run "existing_network_attaches_data_stores" {
  command = plan

  variables {
    network_id            = "projects/host-proj/global/networks/shared"
    create_psa_connection = false
    create_runtime        = false
  }

  assert {
    condition = alltrue([
      length(google_compute_network.this) == 0,
      length(google_compute_subnetwork.this) == 0,
      length(google_compute_global_address.psa) == 0,
      length(google_service_networking_connection.psa) == 0,
      google_sql_database_instance.writer.settings[0].ip_configuration[0].private_network == var.network_id,
      google_redis_instance.this.authorized_network == var.network_id,
    ])
    error_message = "An existing VPC must receive the Cloud SQL and Memorystore private-network attachments."
  }
}

run "psa_required_without_existing_network" {
  command = plan

  variables {
    create_psa_connection = false
  }

  expect_failures = [
    google_sql_database_instance.writer,
  ]
}

run "redis_plaintext_drops_tls_env" {
  command = plan

  variables {
    redis_transit_encryption = false
  }

  assert {
    condition     = google_redis_instance.this.transit_encryption_mode == "DISABLED"
    error_message = "Redis transit encryption must be disabled when requested."
  }

  assert {
    condition     = length(local.shared_env_kv) == 9
    error_message = "Plaintext Redis mode must include GCS and omit the three Redis TLS entries."
  }

  assert {
    condition     = length([for env in local.shared_env_kv : env if env.name == "REDIS_SSL"]) == 0
    error_message = "Plaintext Redis mode must not set REDIS_SSL."
  }

  assert {
    condition     = length(local.redis_ca_fragment) == 0
    error_message = "Plaintext Redis mode must not decode a Redis CA at startup."
  }
}
