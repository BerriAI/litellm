# Plan-only coverage for the gateway request and token autoscaling policies.
# Offline via mock_provider, same as byo_infrastructure.tftest.hcl.

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

run "defaults_scale_on_cpu_and_memory_only" {
  command = plan

  assert {
    condition = alltrue([
      length(aws_appautoscaling_policy.gateway_cpu) == 1,
      length(aws_appautoscaling_policy.gateway_memory) == 1,
      length(aws_appautoscaling_policy.gateway_requests) == 0,
      length(aws_appautoscaling_policy.gateway_tokens) == 0,
    ])
    error_message = "Request and token policies must be absent by default while the CPU and memory policies stay."
  }
}

run "requests_per_second_adds_an_alb_request_count_policy" {
  command = plan

  variables {
    gateway_target_requests_per_second = 90
  }

  assert {
    condition     = length(aws_appautoscaling_policy.gateway_requests) == 1 && length(aws_appautoscaling_policy.gateway_tokens) == 0
    error_message = "A request target alone must add exactly the request policy."
  }

  assert {
    condition = alltrue([
      aws_appautoscaling_policy.gateway_requests[0].name == "acme-litellm-test-gateway-requests",
      aws_appautoscaling_policy.gateway_requests[0].policy_type == "TargetTrackingScaling",
      aws_appautoscaling_policy.gateway_requests[0].service_namespace == "ecs",
      aws_appautoscaling_policy.gateway_requests[0].resource_id == "service/acme-litellm-test/acme-litellm-test-gateway",
      aws_appautoscaling_policy.gateway_requests[0].scalable_dimension == "ecs:service:DesiredCount",
    ])
    error_message = "The request policy must be a target-tracking policy on the gateway service's desired count."
  }

  assert {
    condition = alltrue([
      one(aws_appautoscaling_policy.gateway_requests[0].target_tracking_scaling_policy_configuration).target_value == 5400,
      one(one(aws_appautoscaling_policy.gateway_requests[0].target_tracking_scaling_policy_configuration).predefined_metric_specification).predefined_metric_type == "ALBRequestCountPerTarget",
      length(one(aws_appautoscaling_policy.gateway_requests[0].target_tracking_scaling_policy_configuration).customized_metric_specification) == 0,
    ])
    error_message = "The request policy must track ALBRequestCountPerTarget at 60 times the configured requests per second per task."
  }
}

run "tokens_per_second_adds_a_metric_math_policy" {
  command = plan

  variables {
    gateway_target_tokens_per_second = 6000000
    gateway_tokens_metric = {
      namespace  = "LiteLLM/Prometheus"
      dimensions = { ClusterName = "acme-litellm-test", TaskDefinitionFamily = "acme-litellm-test-gateway" }
    }
  }

  assert {
    condition     = length(aws_appautoscaling_policy.gateway_tokens) == 1 && length(aws_appautoscaling_policy.gateway_requests) == 0
    error_message = "A token target alone must add exactly the token policy."
  }

  assert {
    condition = alltrue([
      aws_appautoscaling_policy.gateway_tokens[0].name == "acme-litellm-test-gateway-tokens",
      aws_appautoscaling_policy.gateway_tokens[0].resource_id == "service/acme-litellm-test/acme-litellm-test-gateway",
      one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).target_value == 6000000,
      length(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).predefined_metric_specification) == 0,
    ])
    error_message = "The token policy must track a customized metric at the configured tokens per second per task."
  }

  assert {
    condition = alltrue([
      length(one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics) == 4,
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens"].id == "tokens",
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens"].return_data == false,
      one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens"].metric_stat).stat == "Sum",
      one(one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens"].metric_stat).metric).namespace == "LiteLLM/Prometheus",
      one(one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens"].metric_stat).metric).metric_name == "litellm_total_tokens_metric_total",
      { for d in one(one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens"].metric_stat).metric).dimensions : d.name => d.value } == { ClusterName = "acme-litellm-test", TaskDefinitionFamily = "acme-litellm-test-gateway" },
    ])
    error_message = "The first metric must sum the published token counter deltas under the configured namespace and dimensions."
  }

  assert {
    condition = alltrue([
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["running_tasks"].id == "running_tasks",
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["running_tasks"].return_data == false,
      one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["running_tasks"].metric_stat).stat == "Average",
      one(one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["running_tasks"].metric_stat).metric).namespace == "ECS/ContainerInsights",
      one(one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["running_tasks"].metric_stat).metric).metric_name == "RunningTaskCount",
      { for d in one(one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["running_tasks"].metric_stat).metric).dimensions : d.name => d.value } == { ClusterName = "acme-litellm-test", ServiceName = "acme-litellm-test-gateway" },
    ])
    error_message = "The second metric must read the gateway service's Container Insights RunningTaskCount."
  }

  assert {
    condition = alltrue([
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens_per_second"].expression == "tokens / 60",
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens_per_second"].return_data == false,
    ])
    error_message = "The 60s period Sum must be divided by 60 to yield tokens per second."
  }

  assert {
    condition = alltrue([
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens_per_second_per_task"].expression == "tokens_per_second / running_tasks",
      { for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens_per_second_per_task"].return_data == true,
      length([for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m if m.return_data]) == 1,
    ])
    error_message = "Only the per-task tokens per second may return data to the scaling policy."
  }
}

run "tokens_per_second_needs_the_metric_location" {
  command = plan

  variables {
    gateway_target_tokens_per_second = 6000000
  }

  expect_failures = [
    aws_appautoscaling_policy.gateway_tokens,
  ]
}

run "requests_and_tokens_scale_next_to_cpu_and_memory" {
  command = plan

  variables {
    gateway_target_requests_per_second = 90
    gateway_target_tokens_per_second   = 6000000
    gateway_tokens_metric              = { namespace = "LiteLLM/Prometheus" }
  }

  assert {
    condition = alltrue([
      length(aws_appautoscaling_policy.gateway_cpu) == 1,
      length(aws_appautoscaling_policy.gateway_memory) == 1,
      length(aws_appautoscaling_policy.gateway_requests) == 1,
      length(aws_appautoscaling_policy.gateway_tokens) == 1,
      one(aws_appautoscaling_policy.gateway_cpu[0].target_tracking_scaling_policy_configuration).target_value == 70,
      one(aws_appautoscaling_policy.gateway_memory[0].target_tracking_scaling_policy_configuration).target_value == 80,
    ])
    error_message = "Workload policies must coexist with the CPU and memory policies at their default targets."
  }

  assert {
    condition     = length(one(one({ for m in one(one(aws_appautoscaling_policy.gateway_tokens[0].target_tracking_scaling_policy_configuration).customized_metric_specification).metrics : m.id => m }["tokens"].metric_stat).metric).dimensions) == 0
    error_message = "Omitting dimensions must query the token metric without any."
  }
}

run "workload_targets_are_ignored_when_autoscaling_is_off" {
  command = plan

  variables {
    gateway_autoscaling_enabled        = false
    gateway_target_requests_per_second = 90
    gateway_target_tokens_per_second   = 6000000
    gateway_tokens_metric              = { namespace = "LiteLLM/Prometheus" }
  }

  assert {
    condition = alltrue([
      length(aws_appautoscaling_target.gateway) == 0,
      length(aws_appautoscaling_policy.gateway_requests) == 0,
      length(aws_appautoscaling_policy.gateway_tokens) == 0,
    ])
    error_message = "Disabling gateway autoscaling must drop the workload policies with the target."
  }
}
