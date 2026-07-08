# Task definition for the dedicated litellm-migrations image. Mirrors the
# pre-install/pre-upgrade Helm hook in helm/litellm/templates/migrations-job.yaml.
#
# The image (built from migrations/Dockerfile) ships with
# `ENTRYPOINT ["python3", "/app/run.py"]`. run.py assembles DATABASE_URL from
# the discrete DATABASE_* env vars (IAM auth here) via DatabaseURLSettings,
# then calls ProxyExtrasDBManager.setup_database() — i.e. `prisma migrate
# deploy` with the v2 resolver and P3005/P3009/P3018 recovery. It does NOT
# read CONFIG_FILE_PATH, the master key, or DISABLE_SCHEMA_UPDATE, so we
# don't pass them.
#
# Invoked automatically by `terraform_data.migration` in bootstrap.tf during
# every apply (after the IAM-authed user has been created). The
# `migration_run_command` output is preserved for break-glass manual re-runs.
resource "aws_ecs_task_definition" "migrations" {
  family                   = "${local.name}-migrations"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  # Prisma's Node + Rust engine plus the v2 migration resolver routinely
  # peaks well above 1 GiB while applying the schema. 4 GiB gives plenty
  # of headroom; CPU stays low because `prisma migrate deploy` is
  # single-threaded.
  cpu                = 512
  memory             = 4096
  execution_role_arn = aws_iam_role.task_execution.arn
  task_role_arn      = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "migrations"
    image     = var.migrations_image
    essential = true

    # No entryPoint/command override — the image's ENTRYPOINT runs run.py.
    environment = local.shared_env

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.migrations.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "migrations"
      }
    }
  }])

  tags = local.tags
}
