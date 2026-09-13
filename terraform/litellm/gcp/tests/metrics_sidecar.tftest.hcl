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

run "metrics_sidecar_off_by_default" {
  command = plan

  assert {
    condition     = length(google_cloud_run_v2_service.gateway[0].template[0].containers) == 1
    error_message = "The gateway service must run only the gateway container when gateway_metrics_port is null."
  }

  assert {
    condition     = length([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : e if e.name == "PROMETHEUS_MULTIPROC_DIR"]) == 0
    error_message = "PROMETHEUS_MULTIPROC_DIR must not be set when the metrics sidecar is off."
  }

  assert {
    condition     = length(google_cloud_run_v2_service.gateway[0].template[0].volumes) == 0
    error_message = "No shared multiproc or collector config volume must exist when the metrics sidecar is off."
  }

  assert {
    condition = alltrue([
      length(google_secret_manager_secret.metrics_run_monitoring) == 0,
      length(google_secret_manager_secret_version.metrics_run_monitoring) == 0,
      length(google_secret_manager_secret_iam_member.metrics_run_monitoring) == 0,
      length(google_project_iam_member.runtime_metric_writer) == 0,
      length(google_project_iam_member.runtime_log_writer) == 0,
    ])
    error_message = "No RunMonitoring secret or monitoring IAM must be created when the metrics sidecar is off."
  }
}

run "metrics_sidecar_enabled" {
  command = plan

  variables {
    gateway_metrics_port = 4001
  }

  assert {
    condition     = join(",", [for c in google_cloud_run_v2_service.gateway[0].template[0].containers : c.name]) == "gateway,metrics,collector"
    error_message = "gateway_metrics_port must add the metrics and collector sidecars after the gateway container."
  }

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].image == local.gateway_image,
      join(" ", google_cloud_run_v2_service.gateway[0].template[0].containers[1].command) == "python -m litellm.proxy.prometheus_metrics_server",
      join(" ", google_cloud_run_v2_service.gateway[0].template[0].containers[1].args) == "--port 4001",
    ])
    error_message = "The metrics sidecar must run the gateway image's prometheus_metrics_server on the configured port."
  }

  assert {
    condition = alltrue([
      length([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : e if e.name == "PROMETHEUS_MULTIPROC_DIR" && e.value == "/tmp/litellm_prometheus_multiproc"]) == 1,
      length([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e if e.name == "PROMETHEUS_MULTIPROC_DIR" && e.value == "/tmp/litellm_prometheus_multiproc"]) == 1,
    ])
    error_message = "Gateway and metrics containers must share PROMETHEUS_MULTIPROC_DIR."
  }

  assert {
    condition = alltrue([
      length([for m in google_cloud_run_v2_service.gateway[0].template[0].containers[0].volume_mounts : m if m.name == "prometheus-multiproc" && m.mount_path == "/tmp/litellm_prometheus_multiproc"]) == 1,
      length([for m in google_cloud_run_v2_service.gateway[0].template[0].containers[1].volume_mounts : m if m.name == "prometheus-multiproc" && m.mount_path == "/tmp/litellm_prometheus_multiproc"]) == 1,
      length([for v in google_cloud_run_v2_service.gateway[0].template[0].volumes : v if v.name == "prometheus-multiproc" && length(v.empty_dir) == 1 && v.empty_dir[0].medium == "MEMORY"]) == 1,
    ])
    error_message = "Gateway and metrics containers must mount the same in-memory empty_dir at the multiproc dir."
  }

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].startup_probe[0].http_get[0].path == "/health",
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].startup_probe[0].http_get[0].port == 4001,
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].liveness_probe[0].http_get[0].path == "/health",
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].liveness_probe[0].http_get[0].port == 4001,
    ])
    error_message = "The metrics sidecar must be probed on /health at the configured port."
  }

  assert {
    condition     = length(google_cloud_run_v2_service.gateway[0].template[0].containers[1].ports) == 0 && length(google_cloud_run_v2_service.gateway[0].template[0].containers[2].ports) == 0
    error_message = "Only the gateway container may declare a port; Cloud Run routes ingress to exactly one container."
  }

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.gateway[0].template[0].containers[0].ports[0].container_port == 4000,
      google_cloud_run_v2_service.gateway[0].template[0].containers[0].startup_probe[0].http_get[0].port == 4000,
      google_cloud_run_v2_service.gateway[0].template[0].containers[0].liveness_probe[0].http_get[0].port == 4000,
      google_compute_region_network_endpoint_group.gateway[0].cloud_run[0].service == "${local.name}-gateway",
    ])
    error_message = "The gateway must stay on port 4000 and remain the load balancer's Cloud Run target."
  }

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.gateway[0].template[0].containers[2].image == var.gateway_metrics_collector_image,
      join(",", google_cloud_run_v2_service.gateway[0].template[0].containers[2].depends_on) == "metrics",
      length([for m in google_cloud_run_v2_service.gateway[0].template[0].containers[2].volume_mounts : m if m.name == "gmp-config" && m.mount_path == "/etc/rungmp"]) == 1,
      google_cloud_run_v2_service.gateway[0].template[0].containers[2].liveness_probe[0].http_get[0].port == 13133,
    ])
    error_message = "The collector sidecar must start after the metrics server and read its RunMonitoring config from /etc/rungmp."
  }

  assert {
    condition = alltrue([
      length([for v in google_cloud_run_v2_service.gateway[0].template[0].volumes : v if v.name == "gmp-config" && length(v.secret) == 1 && v.secret[0].items[0].path == "config.yaml"]) == 1,
      google_secret_manager_secret.metrics_run_monitoring[0].secret_id == "${local.name}-gateway-run-monitoring",
      google_secret_manager_secret_iam_member.metrics_run_monitoring[0].role == "roles/secretmanager.secretAccessor",
    ])
    error_message = "The RunMonitoring config must be mounted from a Secret Manager secret readable by the runtime SA."
  }

  assert {
    condition = alltrue([
      yamldecode(google_secret_manager_secret_version.metrics_run_monitoring[0].secret_data).kind == "RunMonitoring",
      yamldecode(google_secret_manager_secret_version.metrics_run_monitoring[0].secret_data).spec.endpoints[0].port == 4001,
      yamldecode(google_secret_manager_secret_version.metrics_run_monitoring[0].secret_data).spec.endpoints[0].path == "/metrics",
    ])
    error_message = "The RunMonitoring config must scrape /metrics on the configured metrics port."
  }

  assert {
    condition = alltrue([
      google_project_iam_member.runtime_metric_writer[0].role == "roles/monitoring.metricWriter",
      google_project_iam_member.runtime_log_writer[0].role == "roles/logging.logWriter",
      google_project_iam_member.runtime_metric_writer[0].project == "test-project",
    ])
    error_message = "The runtime SA must be able to write metrics and logs for the collector sidecar."
  }
}

run "metrics_sidecar_ignored_in_deps_only" {
  command = plan

  variables {
    create_runtime       = false
    gateway_metrics_port = 4001
  }

  assert {
    condition = alltrue([
      length(google_secret_manager_secret.metrics_run_monitoring) == 0,
      length(google_project_iam_member.runtime_metric_writer) == 0,
    ])
    error_message = "Dependencies-only mode must not create metrics sidecar resources."
  }
}

run "metrics_port_rejects_gateway_port" {
  command = plan

  variables {
    gateway_metrics_port = 4000
  }

  expect_failures = [var.gateway_metrics_port]
}

run "metrics_port_rejects_collector_health_port" {
  command = plan

  variables {
    gateway_metrics_port = 13133
  }

  expect_failures = [var.gateway_metrics_port]
}

run "metrics_port_rejects_fractional_port" {
  command = plan

  variables {
    gateway_metrics_port = 4000.5
  }

  expect_failures = [var.gateway_metrics_port]
}
