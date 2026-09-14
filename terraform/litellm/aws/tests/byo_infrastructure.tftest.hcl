# Plan-only coverage for the four networking/database/cache permutations.
# `mock_provider` keeps this offline: no AWS credentials, no API calls, no
# resources. Run from terraform/litellm/aws with `terraform test`.

mock_provider "aws" {
  # IAM policy documents are validated as JSON by the provider, so the
  # generated placeholder string has to be replaced with a parsable one.
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
}

run "module_owns_everything_by_default" {
  command = plan

  variables {
    azs = ["us-east-1a", "us-east-1b"]
  }

  assert {
    condition     = length(aws_vpc.this) == 1 && length(aws_nat_gateway.this) == 1 && length(aws_subnet.private) == 2
    error_message = "The default path must still create its own VPC, NAT gateway, and one private subnet per AZ."
  }

  assert {
    condition     = length(aws_rds_cluster.this) == 1 && length(aws_elasticache_replication_group.this) == 1
    error_message = "The default path must still create Aurora and ElastiCache."
  }

  assert {
    condition     = length(aws_secretsmanager_secret.database_url) == 0 && length(aws_secretsmanager_secret.redis_url) == 0
    error_message = "Connection-string secrets belong to the bring-your-own path only."
  }

  assert {
    condition     = length(local.managed_db_env) == 7 && length(local.managed_redis_env) == 3
    error_message = "Gateway, backend, and migration tasks must keep the discrete DATABASE_*/REDIS_* env for the module-created stores."
  }

  assert {
    condition     = length(terraform_data.bootstrap_db) == 1 && length(aws_ecs_task_definition.migrations) == 1
    error_message = "The IAM-user bootstrap and the schema migration must both run against the module-created Aurora."
  }
}

run "existing_vpc_creates_no_networking" {
  command = plan

  variables {
    vpc_id                             = "vpc-00000000000000001"
    public_subnet_ids                  = ["subnet-pub-a", "subnet-pub-b"]
    private_subnet_ids                 = ["subnet-priv-a", "subnet-priv-b"]
    additional_task_security_group_ids = ["sg-caller-owned"]
  }

  assert {
    condition = alltrue([
      length(aws_vpc.this) == 0,
      length(aws_subnet.public) == 0,
      length(aws_subnet.private) == 0,
      length(aws_internet_gateway.this) == 0,
      length(aws_nat_gateway.this) == 0,
      length(aws_eip.nat) == 0,
      length(aws_route_table.public) == 0,
      length(aws_route_table.private) == 0,
    ])
    error_message = "An existing vpc_id must suppress every network resource, including the route tables and NAT gateway."
  }

  assert {
    condition     = aws_lb.this.subnets == toset(var.public_subnet_ids)
    error_message = "The ALB must land in the caller's public subnets."
  }

  assert {
    condition = alltrue([
      aws_db_subnet_group.this[0].subnet_ids == toset(var.private_subnet_ids),
      aws_elasticache_subnet_group.this[0].subnet_ids == toset(var.private_subnet_ids),
      aws_ecs_service.gateway.network_configuration[0].subnets == toset(var.private_subnet_ids),
    ])
    error_message = "Tasks, Aurora, and ElastiCache must land in the caller's private subnets."
  }

  assert {
    condition     = length(local.task_security_group_ids) == 2
    error_message = "additional_task_security_group_ids must be attached alongside the module's own tasks group."
  }
}

run "existing_database_and_redis_replace_the_managed_ones" {
  command = plan

  variables {
    azs             = ["us-east-1a", "us-east-1b"]
    create_database = false
    database_url    = "postgresql://litellm:pw@db.internal:5432/litellm"
    create_redis    = false
    redis_url       = "rediss://:pw@cache.internal:6379"
  }

  assert {
    condition = alltrue([
      length(aws_rds_cluster.this) == 0,
      length(aws_rds_cluster_instance.writer) == 0,
      length(aws_db_subnet_group.this) == 0,
      length(aws_security_group.rds) == 0,
      length(aws_elasticache_replication_group.this) == 0,
      length(aws_elasticache_subnet_group.this) == 0,
      length(aws_security_group.redis) == 0,
    ])
    error_message = "Pointing at an existing database and cache must create neither Aurora nor ElastiCache."
  }

  assert {
    condition     = length(local.managed_db_env) == 0 && length(local.managed_redis_env) == 0
    error_message = "The discrete DATABASE_*/REDIS_* env vars must be dropped so DATABASE_URL/REDIS_URL are the only connection targets."
  }

  assert {
    condition = alltrue([
      length([for s in local.shared_secrets : s if s.name == "DATABASE_URL"]) == 1,
      length([for s in local.shared_secrets : s if s.name == "REDIS_URL"]) == 1,
    ])
    error_message = "Both connection strings must reach the containers as Secrets Manager references, not plain-text env."
  }

  assert {
    condition     = length(terraform_data.bootstrap_db) == 0 && length(aws_ecs_task_definition.migrations) == 1
    error_message = "An existing database still needs the schema migration, but not the Aurora IAM-user bootstrap."
  }

  assert {
    condition     = length([for e in local.backend_default_env : e if e.name == "STORE_MODEL_IN_DB"]) == 1
    error_message = "STORE_MODEL_IN_DB must stay set when a database is reachable."
  }
}

run "vpc_without_subnets_fails_at_plan" {
  command = plan

  variables {
    vpc_id = "vpc-00000000000000001"
  }

  expect_failures = [
    aws_lb.this,
    aws_security_group.tasks,
  ]
}

run "neither_vpc_nor_azs_fails_at_plan" {
  command = plan

  expect_failures = [
    aws_vpc.this,
  ]
}

# Aurora and ElastiCache subnet groups need two AZs, so one private subnet is
# only enough when neither store is module-created.
run "one_private_subnet_fails_while_a_managed_store_needs_two_azs" {
  command = plan

  variables {
    vpc_id             = "vpc-00000000000000001"
    public_subnet_ids  = ["subnet-pub-a", "subnet-pub-b"]
    private_subnet_ids = ["subnet-priv-a"]
  }

  expect_failures = [
    aws_security_group.tasks,
  ]
}

run "one_private_subnet_is_enough_without_managed_stores" {
  command = plan

  variables {
    vpc_id             = "vpc-00000000000000001"
    public_subnet_ids  = ["subnet-pub-a", "subnet-pub-b"]
    private_subnet_ids = ["subnet-priv-a"]
    create_database    = false
    create_redis       = false
    # Single process, so the Redis-less rate-limit check stays quiet and this
    # run is only exercising the subnet rule.
    gateway_autoscaling_enabled = false
    gateway_desired_count       = 1
    gateway_num_workers         = 1
  }

  assert {
    condition     = length(aws_security_group.tasks.vpc_id) > 0
    error_message = "With no module-created database or cache, a single private subnet must plan cleanly."
  }
}

# The default sizing is 10 tasks under autoscaling, so a Redis-less stack must
# warn that per-key limits are counted per process.
run "redis_less_multi_process_gateway_is_flagged" {
  command = plan

  variables {
    azs          = ["us-east-1a", "us-east-1b"]
    create_redis = false
  }

  expect_failures = [
    check.redis_less_rate_limits_are_per_process,
  ]
}

run "redis_less_single_process_gateway_is_not_flagged" {
  command = plan

  variables {
    azs                         = ["us-east-1a", "us-east-1b"]
    create_redis                = false
    gateway_autoscaling_enabled = false
    gateway_desired_count       = 1
    gateway_num_workers         = 1
  }

  assert {
    condition     = local.max_gateway_processes == 1
    error_message = "One task with one worker is a single process, which is the supported way to run without Redis."
  }
}

run "no_database_and_no_redis_drops_the_schema_migration" {
  command = plan

  variables {
    azs             = ["us-east-1a", "us-east-1b"]
    create_database = false
    create_redis    = false
    # Single process, so the Redis-less rate-limit check stays quiet here; it
    # has its own run above.
    gateway_autoscaling_enabled = false
    gateway_desired_count       = 1
    gateway_num_workers         = 1
  }

  assert {
    condition = alltrue([
      length(aws_ecs_task_definition.migrations) == 0,
      length(terraform_data.migration) == 0,
      length(aws_iam_policy.rds_iam_connect) == 0,
      length(aws_secretsmanager_secret.db_master_password) == 0,
    ])
    error_message = "With no database at all there is nothing to migrate, bootstrap, or grant rds-db:connect on."
  }

  assert {
    condition     = length(local.backend_default_env) == 0
    error_message = "STORE_MODEL_IN_DB must not be set without a database to store models in."
  }

  assert {
    condition     = length(local.shared_env) == 4
    error_message = "The shared env must narrow to the S3 bucket and region pair when both data stores are gone."
  }
}
