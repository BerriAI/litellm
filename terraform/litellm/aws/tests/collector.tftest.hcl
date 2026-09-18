# Plan-only coverage for the opt-in collector sidecar in the gateway task.
# The rendered container_definitions JSON is unknown at plan time (it embeds
# Aurora/ElastiCache endpoints and secret ARNs), so the assertions target the
# locals it is built from. Run from terraform/litellm/aws with `terraform test`.

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
  allow_plaintext_alb = true
  azs                 = ["us-east-1a", "us-east-1b"]
}

run "disabled_by_default_leaves_the_task_untouched" {
  command = plan

  assert {
    condition     = length(local.collector_container) == 0
    error_message = "The gateway task must stay single-container unless collector_enabled is set."
  }

  assert {
    condition     = !anytrue([for e in local.gateway_environment : startswith(e.name, "LITELLM_COLLECTOR_")])
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
    gateway_extra_env        = { OPENAI_API_BASE = "https://example.invalid" }
    gateway_extra_secrets    = { OPENAI_API_KEY = "arn:aws:secretsmanager:us-east-1:111122223333:secret:openai-AbCdEf" }
  }

  assert {
    condition     = length(local.collector_container) == 1 && local.collector_container[0].name == "collector"
    error_message = "Enabling the sidecar must add exactly one collector container."
  }

  assert {
    condition = alltrue([
      for env in [local.gateway_environment, local.collector_container[0].environment] : (
        { for e in env : e.name => e.value }["LITELLM_COLLECTOR_ENABLED"] == "true" &&
        { for e in env : e.name => e.value }["LITELLM_COLLECTOR_ADDRESS"] == "tcp://127.0.0.1:4321" &&
        { for e in env : e.name => e.value }["LITELLM_COLLECTOR_BUFFER_SIZE"] == "250" &&
        { for e in env : e.name => e.value }["LITELLM_COLLECTOR_ON_UNAVAILABLE"] == "drop" &&
        { for e in env : e.name => e.value }["LITELLM_COLLECTOR_DRAIN_TIMEOUT_SECONDS"] == "10"
      )
    ])
    error_message = "Gateway and sidecar must agree on the loopback address and the collector knobs."
  }

  assert {
    condition = (
      local.collector_container[0].image == var.gateway_image &&
      local.collector_container[0].entryPoint == ["sh", "-c"] &&
      local.collector_container[0].command == ["exec python -m litellm.proxy.collector"] &&
      local.collector_container[0].essential == false &&
      local.collector_container[0].restartPolicy.enabled == true &&
      { for e in local.collector_container[0].environment : e.name => e.value }["LITELLM_JOB_ROLE"] == "collector"
    )
    error_message = "The sidecar must run litellm.proxy.collector from the gateway image as a restartable, non-essential collector."
  }

  assert {
    condition = (
      { for e in local.collector_container[0].environment : e.name => e.value }["OPENAI_API_BASE"] == "https://example.invalid" &&
      contains([for e in local.collector_container[0].environment : e.name], "DATABASE_HOST") &&
      contains([for e in local.collector_container[0].environment : e.name], "REDIS_HOST") &&
      contains([for s in local.collector_container[0].secrets : s.name], "LITELLM_MASTER_KEY") &&
      contains([for s in local.collector_container[0].secrets : s.name], "OPENAI_API_KEY")
    )
    error_message = "The sidecar must receive the gateway's database, Redis, and shared secrets plus gateway_extra_env / gateway_extra_secrets."
  }

  assert {
    condition     = !contains(keys(local.collector_container[0]), "portMappings")
    error_message = "The sidecar must not expose a port to the task's load balancer."
  }

  assert {
    condition     = local.collector_container[0].cpu == 512 && local.collector_container[0].memory == 2048
    error_message = "The sidecar defaults must mirror helm's collector resources (500m / 2Gi)."
  }
}

run "proxy_config_is_fetched_by_the_sidecar_too" {
  command = plan

  variables {
    collector_enabled = true
    proxy_config      = { model_list = [] }
  }

  assert {
    condition = (
      startswith(local.collector_container[0].command[0], local.proxy_config_fetch_cmd) &&
      endswith(local.collector_container[0].command[0], "exec python -m litellm.proxy.collector") &&
      contains([for e in local.collector_container[0].environment : e.name], "CONFIG_FILE_PATH")
    )
    error_message = "The sidecar must pull the proxy config from S3 before starting, like the gateway does."
  }
}

run "sidecar_must_leave_room_for_the_gateway" {
  command = plan

  variables {
    collector_enabled = true
    collector_cpu     = 1024
  }

  expect_failures = [
    aws_ecs_task_definition.gateway,
  ]
}

run "sidecars_must_not_share_a_loopback_port" {
  command = plan

  variables {
    collector_enabled    = true
    collector_port       = 4001
    gateway_metrics_port = 4001
  }

  expect_failures = [
    aws_ecs_task_definition.gateway,
  ]
}
