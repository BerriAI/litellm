# Plan-only coverage for the Prometheus metrics sidecar wiring. Offline via
# mock_provider, same as byo_infrastructure.tftest.hcl.

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

run "defaults_change_nothing" {
  command = plan

  assert {
    condition = alltrue([
      length(local.gateway_metrics_container) == 0,
      length(local.metrics_env) == 0,
      length(local.metrics_mount_points) == 0,
      length([for r in aws_security_group.tasks.ingress : r if r.description == "Prometheus scrapers to the gateway metrics sidecar"]) == 0,
    ])
    error_message = "The metrics sidecar, its env, its volume, and its security-group rule must all be absent by default."
  }
}

run "metrics_port_adds_a_sidecar_volume_and_scrape_rule" {
  command = plan

  variables {
    gateway_metrics_port         = 9464
    gateway_metrics_scrape_cidrs = ["10.20.0.0/16"]
  }

  assert {
    condition     = length(local.metrics_env) == 1 && local.metrics_env[0].name == "PROMETHEUS_MULTIPROC_DIR" && local.metrics_env[0].value == "/tmp/litellm_prometheus_multiproc"
    error_message = "The gateway workers must write multiprocess samples to the shared dir."
  }

  assert {
    condition     = length(local.metrics_mount_points) == 1 && local.metrics_mount_points[0].sourceVolume == "prometheus-multiproc" && local.metrics_mount_points[0].containerPath == "/tmp/litellm_prometheus_multiproc"
    error_message = "Gateway and sidecar must mount the same task volume at the multiproc dir."
  }

  assert {
    condition = alltrue([
      length(local.gateway_metrics_container) == 1,
      local.gateway_metrics_container[0].name == "metrics",
      local.gateway_metrics_container[0].essential == false,
      join(" ", local.gateway_metrics_container[0].entryPoint) == "python -m litellm.proxy.prometheus_metrics_server",
      join(" ", local.gateway_metrics_container[0].command) == "--port 9464",
      one(local.gateway_metrics_container[0].portMappings).containerPort == 9464,
      one(local.gateway_metrics_container[0].environment).value == "/tmp/litellm_prometheus_multiproc",
      one(local.gateway_metrics_container[0].mountPoints).sourceVolume == "prometheus-multiproc",
      strcontains(local.gateway_metrics_container[0].healthCheck.command[3], "9464"),
    ])
    error_message = "The metrics sidecar must run prometheus_metrics_server on the configured port, share the multiproc volume, and health-check that port."
  }

  assert {
    condition     = length(aws_ecs_task_definition.gateway.volume) == 1 && one(aws_ecs_task_definition.gateway.volume).name == "prometheus-multiproc"
    error_message = "The gateway task must declare the multiproc volume."
  }

  assert {
    condition = length([
      for r in aws_security_group.tasks.ingress : r
      if r.from_port == 9464 && r.to_port == 9464 && r.protocol == "tcp" && r.cidr_blocks == tolist(["10.20.0.0/16"])
    ]) == 1
    error_message = "The scrape CIDRs must be allowed to reach the metrics port on the tasks security group."
  }

  assert {
    condition     = aws_lb_target_group.gateway.port == 4000 && one(aws_ecs_service.gateway.load_balancer).container_port == 4000
    error_message = "The ALB must keep targeting the gateway port only; the metrics port is never load balanced."
  }
}

run "metrics_port_without_scrape_cidrs_opens_nothing" {
  command = plan

  variables {
    gateway_metrics_port = 9464
  }

  assert {
    condition     = length(local.gateway_metrics_container) == 1 && length([for r in aws_security_group.tasks.ingress : r if r.from_port == 9464]) == 0
    error_message = "Without scrape CIDRs the sidecar runs but the metrics port stays closed to everything but the ALB group."
  }
}

run "metrics_port_may_not_reuse_the_gateway_port" {
  command = plan

  variables {
    gateway_metrics_port = 4000
  }

  expect_failures = [var.gateway_metrics_port]
}
