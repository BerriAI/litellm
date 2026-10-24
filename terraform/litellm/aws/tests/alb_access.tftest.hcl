mock_provider "aws" {
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }
}
mock_provider "random" {}

variables {
  region              = "eu-west-2"
  tenant              = "acme"
  env                 = "test"
  azs                 = ["eu-west-2a", "eu-west-2b"]
  allow_plaintext_alb = true
  image_tag           = "v9.9.9"
}

run "defaults_keep_the_public_alb" {
  command = plan

  assert {
    condition     = aws_lb.this.internal == false
    error_message = "The ALB must stay internet-facing unless alb_internal is set."
  }

  assert {
    condition = alltrue([
      for rule in aws_security_group.alb.ingress : toset(rule.cidr_blocks) == toset(["0.0.0.0/0"])
    ])
    error_message = "The ALB must stay open to everyone unless alb_ingress_cidrs is set."
  }

  assert {
    condition     = aws_lb_listener.http.default_action[0].type == "forward" && length(aws_lb_listener_rule.management) == 0
    error_message = "Without management_source_cidrs the backend must stay the listener's default."
  }

  assert {
    condition     = length(aws_lb_listener_rule.gateway) == ceil(length(local.default_gateway_path_prefixes) / 5)
    error_message = "Without gateway_path_prefixes every built-in gateway route must be forwarded."
  }
}

run "internal_alb_with_narrowed_ingress" {
  command = plan

  variables {
    alb_internal      = true
    alb_ingress_cidrs = ["10.0.0.0/16"]
  }

  assert {
    condition     = aws_lb.this.internal == true
    error_message = "alb_internal must make the ALB internal."
  }

  assert {
    condition = alltrue([
      for rule in aws_security_group.alb.ingress : toset(rule.cidr_blocks) == toset(["10.0.0.0/16"])
    ])
    error_message = "alb_ingress_cidrs must replace 0.0.0.0/0 on both ALB ports."
  }
}

run "management_routes_answer_only_the_source_range" {
  command = plan

  variables {
    management_source_cidrs = ["10.0.5.10/32"]
  }

  assert {
    condition = alltrue([
      aws_lb_listener.http.default_action[0].type == "fixed-response",
      aws_lb_listener.http.default_action[0].fixed_response[0].status_code == "403",
    ])
    error_message = "With management_source_cidrs anything unmatched must get 403, not the backend."
  }

  assert {
    condition = alltrue([
      aws_lb_listener_rule.management[0].action[0].type == "forward",
      aws_lb_listener_rule.management[0].priority > 100 + length(aws_lb_listener_rule.gateway),
      one(aws_lb_listener_rule.management[0].condition).source_ip[0].values == toset(["10.0.5.10/32"]),
    ])
    error_message = "The backend must be reached only from management_source_cidrs, after the gateway rules."
  }

  assert {
    condition = alltrue([
      for rule in [aws_lb_listener_rule.ui_exact, aws_lb_listener_rule.ui_prefix] :
      length([for c in rule.condition : c if length(c.source_ip) > 0 && c.source_ip[0].values == toset(["10.0.5.10/32"])]) == 1
    ])
    error_message = "The UI must be reached only from management_source_cidrs."
  }
}

run "management_routes_on_the_https_listener" {
  command = plan

  variables {
    allow_plaintext_alb     = false
    acm_certificate_arn     = "arn:aws:acm:eu-west-2:000000000000:certificate/test"
    management_source_cidrs = ["10.0.5.10/32"]
  }

  assert {
    condition = alltrue([
      aws_lb_listener.http.default_action[0].type == "redirect",
      aws_lb_listener.https[0].default_action[0].type == "fixed-response",
      aws_lb_listener.https[0].default_action[0].fixed_response[0].status_code == "403",
    ])
    error_message = "With TLS the 80 listener must still redirect and the 443 listener must answer 403 by default."
  }
}

run "more_than_one_management_cidr_is_refused" {
  command = plan

  variables {
    management_source_cidrs = ["10.0.5.10/32", "10.0.6.10/32"]
  }

  expect_failures = [var.management_source_cidrs]
}

run "gateway_routes_can_be_narrowed" {
  command = plan

  variables {
    gateway_path_prefixes = ["/v1/chat/*", "/chat/*", "/v1/embeddings*", "/embeddings*", "/v1/models*", "/models*"]
  }

  assert {
    condition = alltrue([
      length(aws_lb_listener_rule.gateway) == 2,
      toset(flatten([for r in aws_lb_listener_rule.gateway : one(r.condition).path_pattern[0].values])) == toset(["/v1/chat/*", "/chat/*", "/v1/embeddings*", "/embeddings*", "/v1/models*", "/models*"]),
    ])
    error_message = "gateway_path_prefixes must replace the built-in route list."
  }
}
