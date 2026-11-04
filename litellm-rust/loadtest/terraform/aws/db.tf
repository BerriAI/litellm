resource "random_password" "db" {
  count   = var.create_postgres ? 1 : 0
  length  = 32
  special = false
}

resource "aws_db_subnet_group" "keys" {
  count      = var.create_postgres ? 1 : 0
  name       = var.name
  subnet_ids = [aws_subnet.main.id, aws_subnet.db_secondary.id]
}

resource "aws_db_instance" "keys" {
  count                  = var.create_postgres ? 1 : 0
  identifier             = var.name
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = var.postgres_instance_class
  allocated_storage      = 50
  storage_type           = "gp3"
  db_name                = "litellm"
  username               = "litellm"
  password               = random_password.db[0].result
  db_subnet_group_name   = aws_db_subnet_group.keys[0].name
  vpc_security_group_ids = [aws_security_group.rig.id]
  availability_zone      = local.az
  skip_final_snapshot    = true
  apply_immediately      = true
}

locals {
  database_url = var.create_postgres ? "postgresql://litellm:${random_password.db[0].result}@${aws_db_instance.keys[0].address}:5432/litellm" : ""
}
