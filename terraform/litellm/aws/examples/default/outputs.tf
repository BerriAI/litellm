output "alb_dns_name" {
  description = "Public DNS name of the LiteLLM ALB."
  value       = module.litellm.alb_dns_name
}

output "alb_url" {
  description = "Proxy URL. Dashboard at /, API at /v1/*."
  value       = module.litellm.alb_url
}

output "ecs_cluster" {
  description = "ECS cluster name."
  value       = module.litellm.ecs_cluster
}

output "aurora_writer_endpoint" {
  description = "Aurora writer endpoint."
  value       = module.litellm.aurora_writer_endpoint
}

output "aurora_reader_endpoint" {
  description = "Aurora reader endpoint."
  value       = module.litellm.aurora_reader_endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis primary endpoint (TLS)."
  value       = module.litellm.redis_endpoint
}

output "s3_bucket" {
  description = "S3 bucket name."
  value       = module.litellm.s3_bucket
}

output "master_key_secret_arn" {
  description = "Secrets Manager ARN holding LITELLM_MASTER_KEY."
  value       = module.litellm.master_key_secret_arn
}

output "db_master_password_secret_arn" {
  description = "Secrets Manager ARN holding the Aurora master credentials (bootstrap-only)."
  value       = module.litellm.db_master_password_secret_arn
}

output "db_bootstrap_sql" {
  description = "Run once as the master DB user to create the IAM-authed app user."
  value       = module.litellm.db_bootstrap_sql
}

output "migration_run_command" {
  description = "Break-glass command to re-run the one-off prisma migration task."
  value       = module.litellm.migration_run_command
}
