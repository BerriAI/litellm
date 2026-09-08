# Networking is created only when the caller didn't supply a VPC. With
# var.vpc_id set, every resource in this file except the security groups has
# zero instances and the stack consumes the caller's subnets through
# local.public_subnet_ids / local.private_subnet_ids (see locals.tf).

resource "aws_vpc" "this" {
  count                = local.create_vpc ? 1 : 0
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  lifecycle {
    precondition {
      condition     = length(var.azs) >= 2
      error_message = "Provide at least 2 availability zones in `azs`, or set `vpc_id` + `public_subnet_ids` + `private_subnet_ids` to deploy into an existing VPC."
    }
  }

  tags = merge(local.tags, { Name = local.name })
}

resource "aws_internet_gateway" "this" {
  count  = local.create_vpc ? 1 : 0
  vpc_id = aws_vpc.this[0].id
  tags   = merge(local.tags, { Name = local.name })
}

# Public subnets (ALB + NAT). One per AZ.
resource "aws_subnet" "public" {
  count                   = local.create_vpc ? length(var.azs) : 0
  vpc_id                  = aws_vpc.this[0].id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.tags, { Name = "${local.name}-public-${var.azs[count.index]}" })
}

# Private subnets (ECS tasks, RDS, ElastiCache). One per AZ, separate from
# public range.
resource "aws_subnet" "private" {
  count             = local.create_vpc ? length(var.azs) : 0
  vpc_id            = aws_vpc.this[0].id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = var.azs[count.index]

  tags = merge(local.tags, { Name = "${local.name}-private-${var.azs[count.index]}" })
}

resource "aws_eip" "nat" {
  count  = local.create_vpc ? 1 : 0
  domain = "vpc"
  tags   = merge(local.tags, { Name = "${local.name}-nat" })

  depends_on = [aws_internet_gateway.this]
}

# Single NAT gateway in the first public subnet. For HA, replicate per AZ —
# adds ~$30/mo per gateway, so off by default for a baseline deployment.
resource "aws_nat_gateway" "this" {
  count         = local.create_vpc ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id

  tags = merge(local.tags, { Name = local.name })

  depends_on = [aws_internet_gateway.this]
}

resource "aws_route_table" "public" {
  count  = local.create_vpc ? 1 : 0
  vpc_id = aws_vpc.this[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this[0].id
  }

  tags = merge(local.tags, { Name = "${local.name}-public" })
}

resource "aws_route_table_association" "public" {
  count          = local.create_vpc ? length(var.azs) : 0
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public[0].id
}

resource "aws_route_table" "private" {
  count  = local.create_vpc ? 1 : 0
  vpc_id = aws_vpc.this[0].id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this[0].id
  }

  tags = merge(local.tags, { Name = "${local.name}-private" })
}

resource "aws_route_table_association" "private" {
  count          = local.create_vpc ? length(var.azs) : 0
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[0].id
}

# ---------- Security groups ----------
#
# Always module-owned, in local.vpc_id, so the stack keeps a least-privilege
# path between its own components even when it borrows someone else's VPC.
# Existing databases and caches reached over var.database_url / var.redis_url
# need to allow inbound from the tasks group (or from a group passed via
# var.additional_task_security_group_ids).

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Inbound HTTP/HTTPS to the LiteLLM ALB."
  vpc_id      = local.vpc_id

  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_security_group" "tasks" {
  name        = "${local.name}-tasks"
  description = "ECS tasks (gateway/backend/ui)."
  vpc_id      = local.vpc_id

  ingress {
    description     = "ALB to tasks"
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "All egress (LLM providers, RDS, Redis)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # The tasks group is created in every mode, so this is where the
  # bring-your-own-VPC inputs get checked.
  lifecycle {
    precondition {
      condition     = local.create_vpc || length(var.private_subnet_ids) >= (local.managed_stores_need_two_azs ? 2 : 1)
      error_message = "`private_subnet_ids` is required when `vpc_id` is set: the tasks, Aurora, and ElastiCache all live in private subnets. Aurora and ElastiCache subnet groups need subnets in at least 2 AZs, so pass 2 unless both `create_database` and `create_redis` are false."
    }
  }

  tags = local.tags
}

resource "aws_security_group" "rds" {
  count       = var.create_database ? 1 : 0
  name        = "${local.name}-rds"
  description = "RDS Postgres - tasks only."
  vpc_id      = local.vpc_id

  ingress {
    description     = "Postgres from ECS tasks"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.tasks.id]
  }

  tags = local.tags
}

resource "aws_security_group" "redis" {
  count       = var.create_redis ? 1 : 0
  name        = "${local.name}-redis"
  description = "ElastiCache Redis - tasks only."
  vpc_id      = local.vpc_id

  ingress {
    description     = "Redis from ECS tasks"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.tasks.id]
  }

  tags = local.tags
}
