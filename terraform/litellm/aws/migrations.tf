# Runs the `migrations` component of var.image: run.py assembles DATABASE_URL
# from the discrete DATABASE_* env vars (IAM auth here) and runs `prisma
# migrate deploy`. It does not read CONFIG_FILE_PATH, the master key, or
# DISABLE_SCHEMA_UPDATE, so we don't pass them. Invoked automatically by
# `terraform_data.migration` in bootstrap.tf during every apply; the
# `migration_run_command` output is preserved for break-glass manual re-runs.
resource "aws_ecs_task_definition" "migrations" {
  count                    = local.database_enabled ? 1 : 0
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
    image     = var.image
    essential = true
    command   = ["migrations"]

    environment = local.shared_env
    secrets     = local.byo_database_secrets

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.migrations[0].name
        awslogs-region        = var.region
        awslogs-stream-prefix = "migrations"
      }
    }
  }])

  tags = local.tags
}
