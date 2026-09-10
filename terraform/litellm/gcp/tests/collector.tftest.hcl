# Plan-only coverage for the opt-in collector sidecar on the gateway Cloud
# Run service. `mock_provider` keeps this offline: no GCP credentials, no API
# calls. Run from terraform/litellm/gcp with `terraform test`.

mock_provider "google" {}
mock_provider "google-beta" {}
mock_provider "random" {}

variables {
  project_id         = "acme-test"
  region             = "us-central1"
  tenant             = "acme"
  env                = "test"
  allow_plaintext_lb = true
}

run "disabled_by_default_leaves_the_service_untouched" {
  command = plan

  assert {
    condition     = [for c in google_cloud_run_v2_service.gateway[0].template[0].containers : c.name] == ["gateway"]
    error_message = "The gateway service must stay single-container unless collector_enabled is set."
  }

  assert {
    condition = !anytrue([
      for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : startswith(e.name, "LITELLM_COLLECTOR_")
    ])
    error_message = "No LITELLM_COLLECTOR_* env may reach the gateway while the sidecar is disabled."
  }
}

run "enabled_adds_a_sidecar_that_shares_the_gateway_transport" {
  command = plan

  variables {
    collector_enabled        = true
    collector_port           = 4321
    collector_buffer_size    = 250
    collector_on_unavailable = "drop"
    collector_cpu            = "500m"
    collector_memory         = "1Gi"
    gateway_extra_env        = { OPENAI_API_BASE = "https://example.invalid" }
    gateway_extra_secrets    = { OPENAI_API_KEY = "projects/acme-test/secrets/openai-api-key" }
  }

  assert {
    condition     = [for c in google_cloud_run_v2_service.gateway[0].template[0].containers : c.name] == ["gateway", "spend-collector"]
    error_message = "Enabling the sidecar must append a spend-collector container after the gateway container."
  }

  assert {
    condition = alltrue([
      for c in google_cloud_run_v2_service.gateway[0].template[0].containers : (
        { for e in c.env : e.name => e.value }["LITELLM_COLLECTOR_ENABLED"] == "true" &&
        { for e in c.env : e.name => e.value }["LITELLM_COLLECTOR_ADDRESS"] == "tcp://127.0.0.1:4321" &&
        { for e in c.env : e.name => e.value }["LITELLM_COLLECTOR_BUFFER_SIZE"] == "250" &&
        { for e in c.env : e.name => e.value }["LITELLM_COLLECTOR_ON_UNAVAILABLE"] == "drop" &&
        { for e in c.env : e.name => e.value }["LITELLM_COLLECTOR_DRAIN_TIMEOUT_SECONDS"] == "10"
      )
    ])
    error_message = "Gateway and sidecar must agree on the loopback address and the collector knobs."
  }

  assert {
    condition = (
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].image == local.gateway_image &&
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].command == tolist(["sh", "-c"]) &&
      endswith(google_cloud_run_v2_service.gateway[0].template[0].containers[1].args[0], " && exec python -m litellm.proxy.collector") &&
      strcontains(google_cloud_run_v2_service.gateway[0].template[0].containers[1].args[0], "export DATABASE_URL=") &&
      strcontains(google_cloud_run_v2_service.gateway[0].template[0].containers[1].args[0], "REDIS_SSL_CA_CERTS") &&
      { for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name => e.value }["LITELLM_JOB_ROLE"] == "collector"
    )
    error_message = "The sidecar must run litellm.proxy.collector from the gateway image with the same Redis CA + DATABASE_URL bootstrap as the gateway."
  }

  assert {
    condition = (
      length(google_cloud_run_v2_service.gateway[0].template[0].containers[1].ports) == 0 &&
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].resources[0].limits.cpu == "500m" &&
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].resources[0].limits.memory == "1Gi"
    )
    error_message = "The sidecar must not claim the ingress port and must carry its own resource limits."
  }

  assert {
    condition = (
      { for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name => e.value }["OPENAI_API_BASE"] == "https://example.invalid" &&
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name], "DATABASE_HOST") &&
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name], "REDIS_HOST") &&
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name if length(e.value_source) > 0], "LITELLM_MASTER_KEY") &&
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name if length(e.value_source) > 0], "DATABASE_PASSWORD") &&
      contains([for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name if length(e.value_source) > 0], "OPENAI_API_KEY")
    )
    error_message = "The sidecar must receive the gateway's database, Redis, and Secret Manager env plus gateway_extra_env / gateway_extra_secrets."
  }
}

run "coexists_with_the_metrics_sidecars" {
  command = plan

  variables {
    collector_enabled    = true
    gateway_metrics_port = 4001
  }

  assert {
    condition     = [for c in google_cloud_run_v2_service.gateway[0].template[0].containers : c.name] == ["gateway", "metrics", "collector", "spend-collector"]
    error_message = "The spend collector must keep its own container name next to the GMP metrics collector."
  }

  assert {
    condition = (
      { for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : e.name => e.value }["PROMETHEUS_MULTIPROC_DIR"] == local.metrics_multiproc_dir &&
      { for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : e.name => e.value }["LITELLM_COLLECTOR_ENABLED"] == "true"
    )
    error_message = "The gateway container must keep both the metrics and the collector env when both sidecars are on."
  }
}

run "sidecars_must_not_share_a_loopback_port" {
  command = plan

  variables {
    collector_enabled    = true
    collector_port       = 4001
    gateway_metrics_port = 4001
  }

  expect_failures = [
    google_cloud_run_v2_service.gateway,
  ]
}

run "collector_cannot_take_the_metrics_sidecar_health_port" {
  command = plan

  variables {
    collector_enabled = true
    collector_port    = 13133
  }

  expect_failures = [
    var.collector_port,
  ]
}

run "proxy_config_is_mounted_into_the_sidecar_too" {
  command = plan

  variables {
    collector_enabled = true
    proxy_config      = { model_list = [] }
  }

  assert {
    condition = alltrue([
      for c in google_cloud_run_v2_service.gateway[0].template[0].containers : (
        [for m in c.volume_mounts : m.name] == [local.proxy_config_volume] &&
        contains([for e in c.env : e.name], "CONFIG_FILE_PATH")
      )
    ])
    error_message = "Both containers must mount the proxy-config GCS volume and point CONFIG_FILE_PATH at it."
  }
}
