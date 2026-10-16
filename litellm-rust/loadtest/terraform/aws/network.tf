data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  # Everything that carries load sits in one AZ and one cluster placement group, so the network
  # between generators, gateway and mocks is not what gets measured.
  az = data.aws_availability_zones.available.names[0]
}

resource "aws_vpc" "this" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  tags                 = { Name = var.name }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
}

resource "aws_subnet" "main" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = "10.42.0.0/20"
  availability_zone       = local.az
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.name}-main" }
}

# RDS needs a subnet group spanning two AZs; only the DB uses this one.
resource "aws_subnet" "db_secondary" {
  vpc_id            = aws_vpc.this.id
  cidr_block        = "10.42.16.0/20"
  availability_zone = data.aws_availability_zones.available.names[1]
  tags              = { Name = "${var.name}-db-secondary" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
}

resource "aws_route_table_association" "main" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "rig" {
  name   = "${var.name}-rig"
  vpc_id = aws_vpc.this.id

  ingress {
    description = "everything inside the rig"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }
  ingress {
    description = "VPC (NLB health checks and traffic)"
    from_port   = 4000
    to_port     = 4000
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.this.cidr_block]
  }
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  ingress {
    description = "gateway, for manual demo calls"
    from_port   = 4000
    to_port     = 4000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_placement_group" "rig" {
  name     = var.name
  strategy = "cluster"
}
