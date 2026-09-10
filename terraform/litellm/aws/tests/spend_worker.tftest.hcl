# Plan-only coverage for the opt-in spend-worker sidecar in the gateway task.
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
    condition     = length(local.spend_worker_container) == 0
    error_message = "The gateway task must stay single-container unless spend_worker_enabled is set."
  }

  assert {
    condition     = !anytrue([for e in local.gateway_environment : startswith(e.name, "LITELLM_SPEND_WORKER_")])
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
    gateway_extra_env           = { OPENAI_API_BASE = "https://example.invalid" }
    gateway_extra_secrets       = { OPENAI_API_KEY = "arn:aws:secretsmanager:us-east-1:111122223333:secret:openai-AbCdEf" }
  }

  assert {
    condition     = length(local.spend_worker_container) == 1 && local.spend_worker_container[0].name == "spend-worker"
    error_message = "Enabling the sidecar must add exactly one spend-worker container."
  }

  assert {
    condition = alltrue([
      for env in [local.gateway_environment, local.spend_worker_container[0].environment] : (
        { for e in env : e.name => e.value }["LITELLM_SPEND_WORKER_ENABLED"] == "true" &&
        { for e in env : e.name => e.value }["LITELLM_SPEND_WORKER_ADDRESS"] == "tcp://127.0.0.1:4321" &&
        { for e in env : e.name => e.value }["LITELLM_SPEND_WORKER_BUFFER_SIZE"] == "250" &&
        { for e in env : e.name => e.value }["LITELLM_SPEND_WORKER_ON_UNAVAILABLE"] == "drop" &&
        { for e in env : e.name => e.value }["LITELLM_SPEND_WORKER_DRAIN_TIMEOUT_SECONDS"] == "10"
      )
    ])
    error_message = "Gateway and sidecar must agree on the loopback address and the spend-worker knobs."
  }

  assert {
    condition = (
      local.spend_worker_container[0].image == var.gateway_image &&
      local.spend_worker_container[0].entryPoint == ["sh", "-c"] &&
      local.spend_worker_container[0].command == ["exec python -m gateway.spend_worker"] &&
      local.spend_worker_container[0].essential == false &&
      local.spend_worker_container[0].restartPolicy.enabled == true &&
      { for e in local.spend_worker_container[0].environment : e.name => e.value }["LITELLM_JOB_ROLE"] == "spend_worker"
    )
    error_message = "The sidecar must run gateway.spend_worker from the gateway image as a restartable, non-essential spend_worker."
  }

  assert {
    condition = (
      { for e in local.spend_worker_container[0].environment : e.name => e.value }["OPENAI_API_BASE"] == "https://example.invalid" &&
      contains([for e in local.spend_worker_container[0].environment : e.name], "DATABASE_HOST") &&
      contains([for e in local.spend_worker_container[0].environment : e.name], "REDIS_HOST") &&
      contains([for s in local.spend_worker_container[0].secrets : s.name], "LITELLM_MASTER_KEY") &&
      contains([for s in local.spend_worker_container[0].secrets : s.name], "OPENAI_API_KEY")
    )
    error_message = "The sidecar must receive the gateway's database, Redis, and shared secrets plus gateway_extra_env / gateway_extra_secrets."
  }

  assert {
    condition     = !contains(keys(local.spend_worker_container[0]), "portMappings")
    error_message = "The sidecar must not expose a port to the task's load balancer."
  }

  assert {
    condition     = local.spend_worker_container[0].cpu == 512 && local.spend_worker_container[0].memory == 2048
    error_message = "The sidecar defaults must mirror helm's spendWorker resources (500m / 2Gi)."
  }
}

run "proxy_config_is_fetched_by_the_sidecar_too" {
  command = plan

  variables {
    spend_worker_enabled = true
    proxy_config         = { model_list = [] }
  }

  assert {
    condition = (
      startswith(local.spend_worker_container[0].command[0], local.proxy_config_fetch_cmd) &&
      endswith(local.spend_worker_container[0].command[0], "exec python -m gateway.spend_worker") &&
      contains([for e in local.spend_worker_container[0].environment : e.name], "CONFIG_FILE_PATH")
    )
    error_message = "The sidecar must pull the proxy config from S3 before starting, like the gateway does."
  }
}

run "sidecar_must_leave_room_for_the_gateway" {
  command = plan

  variables {
    spend_worker_enabled = true
    spend_worker_cpu     = 1024
  }

  expect_failures = [
    aws_ecs_task_definition.gateway,
  ]
}
