output "alb_dns_name" {
  description = "Public DNS name of the LiteLLM ALB."
  value       = aws_lb.this.dns_name
}

output "alb_url" {
  description = "Proxy URL. Switches scheme based on whether acm_certificate_arn is set; the underlying DNS name is the ALB. The dashboard is served at /, the API at /v1/*."
  value       = "${local.tls_enabled ? "https" : "http"}://${aws_lb.this.dns_name}"
}

output "ecs_cluster" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.this.name
}

output "aurora_writer_endpoint" {
  description = "Aurora writer endpoint (cluster endpoint). Used by gateway/backend as DATABASE_HOST."
  value       = aws_rds_cluster.this.endpoint
}

output "aurora_reader_endpoint" {
  description = "Aurora reader endpoint. Used by gateway/backend as DATABASE_HOST_READ_REPLICA."
  value       = aws_rds_cluster.this.reader_endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis primary endpoint (TLS, transit_encryption_enabled = true)."
  value       = "${aws_elasticache_replication_group.this.primary_endpoint_address}:${aws_elasticache_replication_group.this.port}"
}

output "s3_bucket" {
  description = "S3 bucket name. Exposed to gateway + backend as S3_BUCKET_NAME / S3_REGION_NAME. Reference from proxy_config via `os.environ/S3_BUCKET_NAME`."
  value       = aws_s3_bucket.this.bucket
}

output "master_key_secret_arn" {
  description = "Secrets Manager ARN holding LITELLM_MASTER_KEY. Fetch with `aws secretsmanager get-secret-value --secret-id <arn>`."
  value       = aws_secretsmanager_secret.master_key.arn
}

output "db_master_password_secret_arn" {
  description = "Secrets Manager ARN holding the Aurora master credentials (bootstrap-only). Used to create the IAM-authed application user."
  value       = aws_secretsmanager_secret.db_master_password.arn
}

# Pre-baked SQL to run once as the master user, creating the IAM-authed
# application user that gateway/backend/migration tasks will authenticate as.
output "db_bootstrap_sql" {
  description = "Run this once as the master DB user (after the first apply) to create the IAM-authed app user."
  value       = <<-SQL
    CREATE USER ${var.db_username};
    GRANT rds_iam TO ${var.db_username};
    GRANT ALL PRIVILEGES ON DATABASE ${var.db_name} TO ${var.db_username};
    GRANT ALL ON SCHEMA public TO ${var.db_username};
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${var.db_username};
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${var.db_username};
  SQL
}

# Pre-baked command for running the one-off migration task. ECS run-task
# needs the subnet + SG IDs at call time, so we render the full command.
output "migration_run_command" {
  description = "Shell command that runs the one-off prisma migration task against Aurora. Run this once, after the bootstrap SQL above, before sending traffic."
  value = format(
    "aws ecs run-task --cluster %s --launch-type FARGATE --task-definition %s --network-configuration 'awsvpcConfiguration={subnets=[%s],securityGroups=[%s],assignPublicIp=DISABLED}' --region %s",
    aws_ecs_cluster.this.name,
    aws_ecs_task_definition.migrations.arn,
    join(",", aws_subnet.private[*].id),
    aws_security_group.tasks.id,
    var.region,
  )
}
