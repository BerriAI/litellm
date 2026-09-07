# Plan-only coverage for the reliability inputs merging into the config.yaml
# that lands in the proxy_config bucket. Offline via mock_provider, same as
# deps_only.tftest.hcl.

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

run "defaults_upload_no_config" {
  command = plan

  assert {
    condition     = local.proxy_config_enabled == false && length(google_storage_bucket_object.proxy_config) == 0
    error_message = "With every reliability input null and no proxy_config there must be no config.yaml to upload."
  }
}

run "reliability_values_land_in_the_uploaded_config" {
  command = plan

  variables {
    sse_keepalive_ping_interval_seconds = 20
    anthropic_sse_ping_interval_seconds = 10
    enable_pre_call_checks              = true
    proxy_config = {
      litellm_settings = { drop_params = true }
      router_settings  = { routing_strategy = "simple-shuffle" }
    }
  }

  assert {
    condition = google_storage_bucket_object.proxy_config[0].content == yamlencode({
      litellm_settings = {
        anthropic_sse_ping_interval_seconds = 10
        drop_params                         = true
        sse_keepalive_ping_interval_seconds = 20
      }
      router_settings = {
        enable_pre_call_checks = true
        routing_strategy       = "simple-shuffle"
      }
    })
    error_message = "config.yaml must carry the keepalive intervals under litellm_settings and enable_pre_call_checks under router_settings next to the user's own keys."
  }

  assert {
    condition     = contains([for e in local.proxy_config_env : e.name], "PROXY_CONFIG_HASH")
    error_message = "The generated config must roll a new revision through PROXY_CONFIG_HASH."
  }
}

run "reliability_values_alone_produce_a_config" {
  command = plan

  variables {
    enable_pre_call_checks = true
  }

  assert {
    condition     = length(google_storage_bucket_object.proxy_config) == 1 && google_storage_bucket_object.proxy_config[0].content == yamlencode({ router_settings = { enable_pre_call_checks = true } })
    error_message = "A reliability input with an empty proxy_config must still upload a config.yaml so the setting reaches the gateway."
  }
}

run "explicit_proxy_config_keys_win" {
  command = plan

  variables {
    sse_keepalive_ping_interval_seconds = 20
    enable_pre_call_checks              = true
    proxy_config = {
      litellm_settings = { sse_keepalive_ping_interval_seconds = 7 }
      router_settings  = { enable_pre_call_checks = false }
    }
  }

  assert {
    condition = google_storage_bucket_object.proxy_config[0].content == yamlencode({
      litellm_settings = { sse_keepalive_ping_interval_seconds = 7 }
      router_settings  = { enable_pre_call_checks = false }
    })
    error_message = "A key written directly into proxy_config must not be overridden by the typed variable."
  }
}

run "keepalive_interval_is_range_checked" {
  command = plan

  variables {
    sse_keepalive_ping_interval_seconds = 301
    anthropic_sse_ping_interval_seconds = 0
  }

  expect_failures = [
    var.sse_keepalive_ping_interval_seconds,
    var.anthropic_sse_ping_interval_seconds,
  ]
}
