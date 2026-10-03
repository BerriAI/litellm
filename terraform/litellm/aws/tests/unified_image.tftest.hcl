# Plan-only coverage for the single-image contract: every task pulls var.image
# and names its process through the image entrypoint's component argument.
# Gateway, backend and migrations container_definitions are unknown at plan
# time (Aurora/ElastiCache endpoints), so those assertions target the locals
# they are built from; the UI task is fully known and is checked as rendered.

mock_provider "aws" {
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }
}
mock_provider "random" {}

variables {
  region               = "us-east-1"
  tenant               = "acme"
  env                  = "test"
  allow_plaintext_alb  = true
  azs                  = ["us-east-1a", "us-east-1b"]
  image                = "123456789012.dkr.ecr.us-east-1.amazonaws.com/litellm:test"
  gateway_num_workers  = 3
  gateway_metrics_port = 9464
  collector_enabled    = true
}

run "every_container_runs_a_component_of_the_one_image" {
  command = plan

  assert {
    condition = alltrue([
      jsondecode(aws_ecs_task_definition.ui.container_definitions)[0].image == var.image,
      join(" ", jsondecode(aws_ecs_task_definition.ui.container_definitions)[0].command) == "ui",
      !can(jsondecode(aws_ecs_task_definition.ui.container_definitions)[0].entryPoint),
    ])
    error_message = "The UI task must run the ui component of var.image through the image entrypoint."
  }

  assert {
    condition = alltrue([
      join(" ", local.component_overrides.gateway.command) == "gateway --workers 3",
      join(" ", local.component_overrides.backend.command) == "backend",
      join(" ", local.component_overrides.collector.command) == "collector",
      alltrue([for component in keys(local.component_args) : !can(local.component_overrides[component].entryPoint)]),
      local.gateway_metrics_container[0].image == var.image,
      join(" ", local.gateway_metrics_container[0].command) == "metrics --port 9464",
      !can(local.gateway_metrics_container[0].entryPoint),
      local.collector_container[0].image == var.image,
    ])
    error_message = "Gateway, backend and the sidecars must keep the image entrypoint and select their component (plus the gateway worker count) through command."
  }
}

run "proxy_config_wraps_the_fetch_around_the_same_component" {
  command = plan

  variables {
    proxy_config = { model_list = [] }
  }

  assert {
    condition = alltrue([
      for component, args in local.component_args :
      join(" ", local.component_overrides[component].entryPoint) == "sh -c" &&
      startswith(local.component_overrides[component].command[0], local.proxy_config_fetch_cmd) &&
      endswith(local.component_overrides[component].command[0], " && exec /app/docker-entrypoint.sh ${join(" ", args)}")
    ])
    error_message = "With proxy_config, gateway, backend and collector must download the config and then exec the image entrypoint with their component."
  }

  assert {
    condition = alltrue([
      !can(local.gateway_metrics_container[0].entryPoint),
      !can(jsondecode(aws_ecs_task_definition.ui.container_definitions)[0].entryPoint),
    ])
    error_message = "The metrics sidecar and the UI never read the proxy config and must keep the plain image entrypoint."
  }
}
