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
    ])
    error_message = "The default mode must create networking, runtime services, the load balancer, and migrations."
  }

  assert {
    condition     = google_redis_instance.this.transit_encryption_mode == "SERVER_AUTHENTICATION"
    error_message = "Redis transit encryption must remain enabled by default."
  }

  assert {
    condition     = length(local.shared_env_kv) == 12
    error_message = "The default runtime environment must include GCS and the three Redis TLS entries."
  }
}

run "deps_only_creates_no_runtime" {
  command = plan

  variables {
    create_runtime = false
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
      google_secret_manager_secret.master_key.secret_id == "tenant-litellm-test-master-key",
      google_secret_manager_secret.db_password.secret_id == "tenant-litellm-test-db-password",
      google_service_account.runtime.account_id == "tenant-litellm-test-runtime",
    ])
    error_message = "Dependencies-only mode must retain data stores, secrets, and the runtime service account."
  }

  assert {
    condition     = output.lb_url == null && output.migration_run_command == null
    error_message = "Runtime outputs must be null while dependency outputs remain available."
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
