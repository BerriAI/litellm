resource "aws_elasticache_subnet_group" "this" {
  name       = "${local.name}-redis"
  subnet_ids = aws_subnet.private[*].id
}

# Replication group (not aws_elasticache_cluster, which is the
# Memcached / single-node Redis resource and can't be upgraded in-place
# to HA). With redis_num_replicas >= 1 we get automatic_failover_enabled
# + multi_az_enabled; at_rest_encryption_enabled and
# transit_encryption_enabled are on unconditionally so Redis traffic is
# TLS-protected — the proxy connects via the rediss:// scheme thanks to
# REDIS_SSL=true in the shared task env (see ecs.tf).
resource "aws_elasticache_replication_group" "this" {
  replication_group_id = "${local.name}-redis"
  description          = "LiteLLM ElastiCache Redis"

  engine               = "redis"
  engine_version       = "7.1"
  node_type            = var.redis_node_type
  num_cache_clusters   = 1 + var.redis_num_replicas
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.this.name
  security_group_ids = [aws_security_group.redis.id]

  automatic_failover_enabled = var.redis_num_replicas >= 1
  multi_az_enabled           = var.redis_num_replicas >= 1
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  apply_immediately = true
}
