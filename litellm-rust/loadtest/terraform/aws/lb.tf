# With several gateways, generators go through an internal NLB like a customer's clients would.
# A non-streaming call sends nothing for up to 6 minutes (the customer's timeout), so the NLB's
# default 350 s TCP idle timeout would cut it; it is raised to 600 s. An ALB in front (as in
# terraform/litellm/aws, idle_timeout = 120) needs the same: at least 400 s.
resource "aws_lb" "gateway" {
  count              = var.gateway_count > 1 ? 1 : 0
  name               = substr("${var.name}-gw", 0, 32)
  internal           = true
  load_balancer_type = "network"
  subnets            = [aws_subnet.main.id]
  security_groups    = [aws_security_group.rig.id]
}

resource "aws_lb_target_group" "gateway" {
  count       = var.gateway_count > 1 ? 1 : 0
  name        = substr("${var.name}-gw", 0, 32)
  port        = 4000
  protocol    = "TCP"
  vpc_id      = aws_vpc.this.id
  target_type = "instance"
  health_check {
    protocol = "TCP"
    interval = 10
  }
  deregistration_delay = 900
}

resource "aws_lb_target_group_attachment" "gateway" {
  count            = var.gateway_count > 1 ? var.gateway_count : 0
  target_group_arn = aws_lb_target_group.gateway[0].arn
  target_id        = aws_instance.gateway[count.index].id
  port             = 4000
}

resource "aws_lb_listener" "gateway" {
  count                    = var.gateway_count > 1 ? 1 : 0
  load_balancer_arn        = aws_lb.gateway[0].arn
  port                     = 4000
  protocol                 = "TCP"
  tcp_idle_timeout_seconds = 600
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.gateway[0].arn
  }
}
