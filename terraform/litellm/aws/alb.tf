resource "aws_lb" "this" {
  name               = local.name
  load_balancer_type = "application"
  internal           = false
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  idle_timeout = 120
}

locals {
  # When an ACM cert ARN is provided we provision a 443 listener carrying
  # the path-routing rules and downgrade the 80 listener to a redirect.
  tls_enabled        = var.acm_certificate_arn != ""
  rules_listener_arn = local.tls_enabled ? aws_lb_listener.https[0].arn : aws_lb_listener.http.arn
}

# Target groups — one per component. IP target type because Fargate tasks
# are addressed by ENI IP, not instance.

resource "aws_lb_target_group" "gateway" {
  name        = "${local.name}-gateway"
  port        = 4000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.this.id

  health_check {
    path                = "/health/readiness"
    matcher             = "200-299"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  deregistration_delay = 30
}

resource "aws_lb_target_group" "backend" {
  name        = "${local.name}-backend"
  port        = 4001
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.this.id

  health_check {
    path                = "/health/readiness"
    matcher             = "200-299"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  deregistration_delay = 30
}

resource "aws_lb_target_group" "ui" {
  name        = "${local.name}-ui"
  port        = 3000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.this.id

  health_check {
    path                = "/healthz"
    matcher             = "200-299"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  deregistration_delay = 30
}

# HTTP listener. When TLS is enabled this only serves a permanent
# 301 redirect to HTTPS; otherwise it carries the path-routing rules
# (default → backend).
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = local.tls_enabled ? "redirect" : "forward"

    dynamic "redirect" {
      for_each = local.tls_enabled ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }

    target_group_arn = local.tls_enabled ? null : aws_lb_target_group.backend.arn
  }

  # Default-deny on the HTTP-only path: TLS is the supported posture.
  # Operators must either supply an ACM cert or explicitly opt in.
  lifecycle {
    precondition {
      condition     = local.tls_enabled || var.allow_plaintext_alb
      error_message = "ALB has no HTTPS listener. Either set `acm_certificate_arn` to enable TLS, or set `allow_plaintext_alb = true` to opt into HTTP-only (trial / dev only)."
    }
  }
}

# HTTPS listener. Only created when an ACM cert ARN is supplied — terminates
# TLS and carries the same default + path-routing rules.
resource "aws_lb_listener" "https" {
  count             = local.tls_enabled ? 1 : 0
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.backend.arn
  }
}

# UI exact paths (/, /favicon.ico, /ui) — priority 10.
resource "aws_lb_listener_rule" "ui_exact" {
  listener_arn = local.rules_listener_arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }

  condition {
    path_pattern {
      values = local.ui_exact_paths
    }
  }
}

# UI prefix paths (/_next/*, /litellm-asset-prefix/*, /assets/*, /ui/*) — priority 20.
resource "aws_lb_listener_rule" "ui_prefix" {
  listener_arn = local.rules_listener_arn
  priority     = 20

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }

  condition {
    path_pattern {
      values = local.ui_path_prefixes
    }
  }
}

# Gateway prefix rules — one per chunk-of-5 because ALB caps a path-pattern
# condition at 5 values. Priorities 100..(100 + N).
resource "aws_lb_listener_rule" "gateway" {
  for_each = { for idx, chunk in local.gateway_path_chunks : idx => chunk }

  listener_arn = local.rules_listener_arn
  priority     = 100 + tonumber(each.key)

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.gateway.arn
  }

  condition {
    path_pattern {
      values = each.value
    }
  }
}
