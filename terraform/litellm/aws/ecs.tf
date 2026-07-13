resource "aws_ecs_cluster" "this" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "gateway" {
  name              = "/ecs/${local.name}/gateway"
  retention_in_days = var.log_retention_days

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${local.name}/backend"
  retention_in_days = var.log_retention_days

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${local.name}/ui"
  retention_in_days = var.log_retention_days

  tags = local.tags
}

resource "aws_cloudwatch_log_group" "migrations" {
  name              = "/ecs/${local.name}/migrations"
  retention_in_days = var.log_retention_days

  tags = local.tags
}

# Shared env block fed to gateway, backend, and the migration task. Mirrors
# the helm chart's `litellm.serverEnv` helper on the IAM-auth branch:
# DATABASE_URL is assembled at runtime by
# litellm/proxy/auth/rds_iam_token.py::init_iam_db_url_from_env from
# HOST/PORT/USER/NAME plus an IAM-signed token, so no DB password is needed
# in the task definition.
locals {
  # OTel v2 is opt-in and gated on otel_endpoint, matching the GCP stack.
  # When set, LITELLM_OTEL_V2 flips on alongside the OTEL_* block, with
  # OTEL_SERVICE_NAME stamped per component so spans land tagged with the
  # right hop. Any OTEL_* key set in *_extra_env wins over the default for
  # that service (ECS allows duplicates but last-wins is undefined, so we
  # filter here for the same predictable behavior GCP gets from Cloud Run's
  # hard duplicate-rejection).
  otel_enabled          = var.otel_endpoint != ""
  otel_environment_name = var.otel_environment_name != "" ? var.otel_environment_name : var.env
  otel_shared_env = local.otel_enabled ? [
    { name = "LITELLM_OTEL_V2", value = "true" },
    { name = "OTEL_EXPORTER", value = var.otel_exporter },
    { name = "OTEL_ENDPOINT", value = var.otel_endpoint },
    { name = "OTEL_ENVIRONMENT_NAME", value = local.otel_environment_name },
    { name = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", value = var.otel_capture_message_content },
  ] : []
  gateway_otel_env_raw = concat(local.otel_shared_env, local.otel_enabled ? [
    { name = "OTEL_SERVICE_NAME", value = "${local.name}-gateway" },
  ] : [])
  backend_otel_env_raw = concat(local.otel_shared_env, local.otel_enabled ? [
    { name = "OTEL_SERVICE_NAME", value = "${local.name}-backend" },
  ] : [])
  gateway_otel_env = [
    for e in local.gateway_otel_env_raw : e if !contains(keys(var.gateway_extra_env), e.name)
  ]
  backend_otel_env = [
    for e in local.backend_otel_env_raw : e if !contains(keys(var.backend_extra_env), e.name)
  ]
  otel_secrets = local.otel_enabled && var.otel_headers_secret_arn != "" ? [
    { name = "OTEL_HEADERS", valueFrom = var.otel_headers_secret_arn },
  ] : []

  shared_env = [
    { name = "IAM_TOKEN_DB_AUTH", value = "true" },
    { name = "DATABASE_HOST", value = aws_rds_cluster.this.endpoint },
    { name = "DATABASE_PORT", value = tostring(aws_rds_cluster.this.port) },
    { name = "DATABASE_USER", value = var.db_username },
    { name = "DATABASE_NAME", value = var.db_name },
    { name = "DATABASE_HOST_READ_REPLICA", value = aws_rds_cluster.this.reader_endpoint },
    { name = "DATABASE_PORT_READ_REPLICA", value = tostring(aws_rds_cluster.this.port) },
    { name = "REDIS_HOST", value = aws_elasticache_replication_group.this.primary_endpoint_address },
    { name = "REDIS_PORT", value = tostring(aws_elasticache_replication_group.this.port) },
    # transit_encryption_enabled = true on the replication group means the
    # proxy must connect via rediss://. _redis.get_redis_url_from_environment
    # honors REDIS_SSL to flip the scheme.
    { name = "REDIS_SSL", value = "true" },
    # S3 bucket — referenced from proxy_config via os.environ/S3_BUCKET_NAME
    # (e.g. cache backend, request log archival, /files passthrough).
    { name = "S3_BUCKET_NAME", value = aws_s3_bucket.this.bucket },
    { name = "S3_REGION_NAME", value = var.region },
    # boto3 inside generate_iam_auth_token reads AWS_REGION_NAME first, then
    # AWS_REGION. Set both for compatibility.
    { name = "AWS_REGION", value = var.region },
    { name = "AWS_REGION_NAME", value = var.region },
  ]

  shared_secrets = concat(
    [
      { name = "LITELLM_MASTER_KEY", valueFrom = aws_secretsmanager_secret.master_key.arn },
    ],
    var.litellm_license == "" ? [] : [
      { name = "LITELLM_LICENSE", valueFrom = aws_secretsmanager_secret.license[0].arn },
    ],
    local.otel_secrets,
  )

  # Backend-only managed secrets. UI_PASSWORD is consumed by the management
  # API (UI login flow) and has no use on the gateway data plane.
  backend_managed_secrets = var.ui_password == "" ? [] : [
    { name = "UI_PASSWORD", valueFrom = aws_secretsmanager_secret.ui_password[0].arn },
  ]

  gateway_extra_env_list = [
    for k, v in var.gateway_extra_env : { name = k, value = v }
  ]
  backend_extra_env_list = [
    for k, v in var.backend_extra_env : { name = k, value = v }
  ]

  backend_default_env = [
    { name = "STORE_MODEL_IN_DB", value = "true" },
  ]
  gateway_extra_secrets_list = [
    for k, v in var.gateway_extra_secrets : { name = k, valueFrom = v }
  ]
  backend_extra_secrets_list = [
    for k, v in var.backend_extra_secrets : { name = k, valueFrom = v }
  ]

  # Mirrors the helm chart's gateway.config.create / configmap pattern.
  # ECS Fargate has no ConfigMap analogue, so the YAML is uploaded to S3
  # (see aws_s3_object.proxy_config in s3.tf) and the container entrypoint
  # downloads it to /tmp/litellm-config.yaml via boto3 before exec'ing
  # uvicorn. The S3 object's etag is embedded in the task definition so a
  # config edit forces a new task-def revision and a rolling redeploy.
  proxy_config_enabled = length(keys(var.proxy_config)) > 0
  proxy_config_path    = "/tmp/litellm-config.yaml"

  proxy_config_env = local.proxy_config_enabled ? [
    { name = "CONFIG_FILE_PATH", value = local.proxy_config_path },
    { name = "LITELLM_PROXY_CONFIG_S3_BUCKET", value = aws_s3_bucket.this.bucket },
    { name = "LITELLM_PROXY_CONFIG_S3_KEY", value = aws_s3_object.proxy_config[0].key },
    { name = "LITELLM_PROXY_CONFIG_S3_ETAG", value = aws_s3_object.proxy_config[0].etag },
  ] : []

  proxy_config_fetch_cmd = "python -c \"import os, boto3; boto3.client('s3', region_name=os.environ['AWS_REGION']).download_file(os.environ['LITELLM_PROXY_CONFIG_S3_BUCKET'], os.environ['LITELLM_PROXY_CONFIG_S3_KEY'], os.environ['CONFIG_FILE_PATH'])\""

  # Gateway always needs --workers wired in (no NUM_WORKERS env var support
  # in the image entrypoint). When proxy_config is enabled we also have to
  # pull the config from S3 first, so the command goes through `sh -c`;
  # otherwise we keep the image's ENTRYPOINT and only override `command`.
  gateway_uvicorn_args = "--host 0.0.0.0 --port 4000 --workers ${var.gateway_num_workers}"
  backend_uvicorn_args = "--host 0.0.0.0 --port 4001"

  gateway_proxy_overrides = local.proxy_config_enabled ? {
    entryPoint = ["sh", "-c"]
    command = [
      "${local.proxy_config_fetch_cmd} && exec uvicorn gateway.main:app ${local.gateway_uvicorn_args}"
    ]
    } : {
    # Mirror the image's ENTRYPOINT so we can append --workers via command.
    entryPoint = ["uvicorn", "gateway.main:app"]
    command    = split(" ", local.gateway_uvicorn_args)
  }

  backend_proxy_overrides = local.proxy_config_enabled ? {
    entryPoint = ["sh", "-c"]
    command = [
      "${local.proxy_config_fetch_cmd} && exec uvicorn backend.main:app ${local.backend_uvicorn_args}"
    ]
  } : {}
}

# ---------- Gateway ----------
resource "aws_ecs_task_definition" "gateway" {
  family                   = "${local.name}-gateway"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.gateway_cpu
  memory                   = var.gateway_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    merge(
      {
        name      = "gateway"
        image     = var.gateway_image
        essential = true

        portMappings = [{ containerPort = 4000, protocol = "tcp" }]
        environment = concat(
          local.shared_env,
          local.gateway_otel_env,
          local.gateway_extra_env_list,
          local.proxy_config_env,
        )
        secrets = concat(local.shared_secrets, local.gateway_extra_secrets_list)

        # Container-level healthCheck intentionally omitted — the wolfi
        # runtime image doesn't ship curl/wget. The ALB target group polls
        # /health/readiness.

        logConfiguration = {
          logDriver = "awslogs"
          options = {
            awslogs-group         = aws_cloudwatch_log_group.gateway.name
            awslogs-region        = var.region
            awslogs-stream-prefix = "gateway"
          }
        }
      },
      local.gateway_proxy_overrides,
    )
  ])

  tags = local.tags
}

resource "aws_ecs_service" "gateway" {
  name            = "${local.name}-gateway"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.gateway.arn
  desired_count   = var.gateway_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.gateway.arn
    container_name   = "gateway"
    container_port   = 4000
  }

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  # desired_count is owned by Application Auto Scaling once enabled (autoscaling.tf).
  # Terraform sets the initial value from var.gateway_desired_count, then steps aside.
  lifecycle {
    ignore_changes = [desired_count]
  }

  # Don't start until the schema migration has run. Otherwise the proxy
  # boots, Prisma fails on the missing tables, and ECS thrashes the task.
  depends_on = [
    aws_lb_listener.http,
    aws_lb_listener.https,
    terraform_data.migration,
  ]

  tags = local.tags
}

# ---------- Backend ----------
resource "aws_ecs_task_definition" "backend" {
  family                   = "${local.name}-backend"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.backend_cpu
  memory                   = var.backend_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    merge(
      {
        name      = "backend"
        image     = var.backend_image
        essential = true

        portMappings = [{ containerPort = 4001, protocol = "tcp" }]
        environment = concat(
          local.shared_env,
          local.backend_default_env,
          local.backend_otel_env,
          local.backend_extra_env_list,
          local.proxy_config_env,
        )
        secrets = concat(local.shared_secrets, local.backend_managed_secrets, local.backend_extra_secrets_list)

        logConfiguration = {
          logDriver = "awslogs"
          options = {
            awslogs-group         = aws_cloudwatch_log_group.backend.name
            awslogs-region        = var.region
            awslogs-stream-prefix = "backend"
          }
        }
      },
      local.backend_proxy_overrides,
    )
  ])

  tags = local.tags
}

resource "aws_ecs_service" "backend" {
  name            = "${local.name}-backend"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = var.backend_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 4001
  }

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [
    aws_lb_listener.http,
    aws_lb_listener.https,
    terraform_data.migration,
  ]

  tags = local.tags
}

# ---------- UI ----------
# task_role is deliberately the unprivileged ui_task — the UI has no DB,
# S3, or Secrets Manager dependency, and inheriting the shared `task`
# role would expose every data-plane secret to a compromised UI
# container via the task metadata endpoint.
resource "aws_ecs_task_definition" "ui" {
  family                   = "${local.name}-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ui_cpu
  memory                   = var.ui_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.ui_task.arn

  container_definitions = jsonencode([
    {
      name         = "ui"
      image        = var.ui_image
      essential    = true
      portMappings = [{ containerPort = 3000, protocol = "tcp" }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ui.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ui"
        }
      }
    }
  ])

  tags = local.tags
}

resource "aws_ecs_service" "ui" {
  name            = "${local.name}-ui"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.ui.arn
  desired_count   = var.ui_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 3000
  }

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [
    aws_lb_listener.http,
    aws_lb_listener.https,
  ]

  tags = local.tags
}
