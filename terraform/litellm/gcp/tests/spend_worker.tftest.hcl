# Plan-only coverage for the opt-in spend-worker sidecar on the gateway Cloud
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
    error_message = "The gateway service must stay single-container unless spend_worker_enabled is set."
  }

  assert {
    condition = !anytrue([
      for e in google_cloud_run_v2_service.gateway[0].template[0].containers[0].env : startswith(e.name, "LITELLM_SPEND_WORKER_")
    ])
    error_message = "No LITELLM_SPEND_WORKER_* env may reach the gateway while the sidecar is disabled."
  }
}

run "enabled_adds_a_sidecar_that_shares_the_gateway_transport" {
  command = plan

  variables {
    spend_worker_enabled        = true
    spend_worker_port           = 4321
    spend_worker_buffer_size    = 250
    spend_worker_on_unavailable = "drop"
    spend_worker_cpu            = "500m"
    spend_worker_memory         = "1Gi"
    gateway_extra_env           = { OPENAI_API_BASE = "https://example.invalid" }
    gateway_extra_secrets       = { OPENAI_API_KEY = "projects/acme-test/secrets/openai-api-key" }
  }

  assert {
    condition     = [for c in google_cloud_run_v2_service.gateway[0].template[0].containers : c.name] == ["gateway", "spend-worker"]
    error_message = "Enabling the sidecar must append a spend-worker container after the gateway container."
  }

  assert {
    condition = alltrue([
      for c in google_cloud_run_v2_service.gateway[0].template[0].containers : (
        { for e in c.env : e.name => e.value }["LITELLM_SPEND_WORKER_ENABLED"] == "true" &&
        { for e in c.env : e.name => e.value }["LITELLM_SPEND_WORKER_ADDRESS"] == "tcp://127.0.0.1:4321" &&
        { for e in c.env : e.name => e.value }["LITELLM_SPEND_WORKER_BUFFER_SIZE"] == "250" &&
        { for e in c.env : e.name => e.value }["LITELLM_SPEND_WORKER_ON_UNAVAILABLE"] == "drop" &&
        { for e in c.env : e.name => e.value }["LITELLM_SPEND_WORKER_DRAIN_TIMEOUT_SECONDS"] == "10"
      )
    ])
    error_message = "Gateway and sidecar must agree on the loopback address and the spend-worker knobs."
  }

  assert {
    condition = (
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].image == local.gateway_image &&
      google_cloud_run_v2_service.gateway[0].template[0].containers[1].command == tolist(["sh", "-c"]) &&
      endswith(google_cloud_run_v2_service.gateway[0].template[0].containers[1].args[0], " && exec python -m gateway.spend_worker") &&
      strcontains(google_cloud_run_v2_service.gateway[0].template[0].containers[1].args[0], "export DATABASE_URL=") &&
      strcontains(google_cloud_run_v2_service.gateway[0].template[0].containers[1].args[0], "REDIS_SSL_CA_CERTS") &&
      { for e in google_cloud_run_v2_service.gateway[0].template[0].containers[1].env : e.name => e.value }["LITELLM_JOB_ROLE"] == "spend_worker"
    )
    error_message = "The sidecar must run gateway.spend_worker from the gateway image with the same Redis CA + DATABASE_URL bootstrap as the gateway."
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

run "proxy_config_is_mounted_into_the_sidecar_too" {
  command = plan

  variables {
    spend_worker_enabled = true
    proxy_config         = { model_list = [] }
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
