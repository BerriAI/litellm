# Application Auto Scaling for the three ECS services. Mirrors the HPA values
# baked into the helm chart at helm/litellm/values.yaml:
#
#   gateway: 1-10 replicas, target 70% CPU + 80% memory
#   backend: 1-4 replicas,  target 70% CPU
#   ui:      1-3 replicas,  target 80% CPU (off by default; nginx static export)
#
# Each service gets a scalable target plus one target-tracking policy per metric.
# When autoscaling is disabled (count=0) the resources collapse cleanly out of
# the plan; the service's desired_count from ecs.tf stays in effect.

# ---------- Gateway ----------
resource "aws_appautoscaling_target" "gateway" {
  count              = var.gateway_autoscaling_enabled ? 1 : 0
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.gateway.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.gateway_min_capacity
  max_capacity       = var.gateway_max_capacity
}

resource "aws_appautoscaling_policy" "gateway_cpu" {
  count              = var.gateway_autoscaling_enabled ? 1 : 0
  name               = "${local.name}-gateway-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.gateway[0].service_namespace
  resource_id        = aws_appautoscaling_target.gateway[0].resource_id
  scalable_dimension = aws_appautoscaling_target.gateway[0].scalable_dimension

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = var.gateway_cpu_target
  }
}

resource "aws_appautoscaling_policy" "gateway_memory" {
  # Memory policy is optional; set gateway_memory_target = 0 to omit it.
  count              = var.gateway_autoscaling_enabled && var.gateway_memory_target > 0 ? 1 : 0
  name               = "${local.name}-gateway-memory"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.gateway[0].service_namespace
  resource_id        = aws_appautoscaling_target.gateway[0].resource_id
  scalable_dimension = aws_appautoscaling_target.gateway[0].scalable_dimension

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = var.gateway_memory_target
  }
}

resource "aws_appautoscaling_policy" "gateway_requests" {
  count              = var.gateway_autoscaling_enabled && var.gateway_target_requests_per_second > 0 ? 1 : 0
  name               = "${local.name}-gateway-requests"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.gateway[0].service_namespace
  resource_id        = aws_appautoscaling_target.gateway[0].resource_id
  scalable_dimension = aws_appautoscaling_target.gateway[0].scalable_dimension

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.this.arn_suffix}/${aws_lb_target_group.gateway.arn_suffix}"
    }
    # ALBRequestCountPerTarget is a per-minute count
    target_value = var.gateway_target_requests_per_second * 60
  }
}

resource "aws_appautoscaling_policy" "gateway_tokens" {
  count              = var.gateway_autoscaling_enabled && var.gateway_target_tokens_per_second > 0 ? 1 : 0
  name               = "${local.name}-gateway-tokens"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.gateway[0].service_namespace
  resource_id        = aws_appautoscaling_target.gateway[0].resource_id
  scalable_dimension = aws_appautoscaling_target.gateway[0].scalable_dimension

  lifecycle {
    precondition {
      condition     = var.gateway_tokens_metric != null
      error_message = "gateway_tokens_metric is required when gateway_target_tokens_per_second > 0."
    }
  }

  target_tracking_scaling_policy_configuration {
    target_value = var.gateway_target_tokens_per_second

    # target tracking has no period setting and always aggregates over 60s
    customized_metric_specification {
      metrics {
        id          = "tokens"
        return_data = false

        metric_stat {
          stat = "Sum"

          metric {
            namespace   = var.gateway_tokens_metric.namespace
            metric_name = var.gateway_tokens_metric.name

            dynamic "dimensions" {
              for_each = var.gateway_tokens_metric.dimensions
              content {
                name  = dimensions.key
                value = dimensions.value
              }
            }
          }
        }
      }

      metrics {
        id          = "running_tasks"
        return_data = false

        metric_stat {
          stat = "Average"

          metric {
            namespace   = "ECS/ContainerInsights"
            metric_name = "RunningTaskCount"

            dimensions {
              name  = "ClusterName"
              value = aws_ecs_cluster.this.name
            }
            dimensions {
              name  = "ServiceName"
              value = aws_ecs_service.gateway.name
            }
          }
        }
      }

      metrics {
        id          = "tokens_per_second"
        expression  = "tokens / 60"
        return_data = false
      }

      metrics {
        id          = "tokens_per_second_per_task"
        expression  = "tokens_per_second / running_tasks"
        label       = "Tokens per second per gateway task"
        return_data = true
      }
    }
  }
}

# ---------- Backend ----------
resource "aws_appautoscaling_target" "backend" {
  count              = var.backend_autoscaling_enabled ? 1 : 0
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.backend.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.backend_min_capacity
  max_capacity       = var.backend_max_capacity
}

resource "aws_appautoscaling_policy" "backend_cpu" {
  count              = var.backend_autoscaling_enabled ? 1 : 0
  name               = "${local.name}-backend-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.backend[0].service_namespace
  resource_id        = aws_appautoscaling_target.backend[0].resource_id
  scalable_dimension = aws_appautoscaling_target.backend[0].scalable_dimension

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = var.backend_cpu_target
  }
}

# ---------- UI ----------
resource "aws_appautoscaling_target" "ui" {
  count              = var.ui_autoscaling_enabled ? 1 : 0
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.ui.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.ui_min_capacity
  max_capacity       = var.ui_max_capacity
}

resource "aws_appautoscaling_policy" "ui_cpu" {
  count              = var.ui_autoscaling_enabled ? 1 : 0
  name               = "${local.name}-ui-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.ui[0].service_namespace
  resource_id        = aws_appautoscaling_target.ui[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ui[0].scalable_dimension

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = var.ui_cpu_target
  }
}
