resource "aws_elasticache_subnet_group" "this" {
  count      = var.create_redis ? 1 : 0
  name       = "${local.name}-redis"
  subnet_ids = local.private_subnet_ids

  tags = local.tags
}

# Replication group (not aws_elasticache_cluster, which is the
# Memcached / single-node Redis resource and can't be upgraded in-place
# to HA). With redis_num_replicas >= 1 we get automatic_failover_enabled
# + multi_az_enabled; at_rest_encryption_enabled and
# transit_encryption_enabled are on unconditionally so Redis traffic is
# TLS-protected — the proxy connects via the rediss:// scheme thanks to
# REDIS_SSL=true in the shared task env (see ecs.tf).
resource "aws_elasticache_replication_group" "this" {
  count                = var.create_redis ? 1 : 0
  replication_group_id = "${local.name}-redis"
  description          = "LiteLLM ElastiCache Redis"

  engine               = "redis"
  engine_version       = "7.1"
  node_type            = var.redis_node_type
  num_cache_clusters   = 1 + var.redis_num_replicas
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.this[0].name
  security_group_ids = [aws_security_group.redis[0].id]

  automatic_failover_enabled = var.redis_num_replicas >= 1
  multi_az_enabled           = var.redis_num_replicas >= 1
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  apply_immediately = true

  tags = local.tags
}

# Rate limits, budgets, and router cooldowns are shared through Redis. Without
# it each gateway process counts on its own, so a caller spread across tasks
# collects the full per-key allowance from every one of them. A `check` rather
# than a precondition: running without Redis is a legitimate choice when you do
# not rely on per-key limits, so this warns instead of blocking the plan.
check "redis_less_rate_limits_are_per_process" {
  assert {
    condition     = local.redis_enabled || local.max_gateway_processes <= 1
    error_message = "No Redis is configured while the gateway can run up to ${local.max_gateway_processes} processes, so per-key RPM/TPM limits, budgets, and cooldowns apply per process and a caller can multiply them across tasks. Set `create_redis = true`, pass `redis_url`, or hold the gateway to one process (`gateway_autoscaling_enabled = false`, `gateway_desired_count = 1`, `gateway_num_workers = 1`)."
  }
}
