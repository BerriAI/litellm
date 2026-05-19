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
