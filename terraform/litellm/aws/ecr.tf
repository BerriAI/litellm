resource "aws_ecr_repository" "gateway" {
  name = "${local.name}-gateway"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_ecr_repository" "backend" {
  name = "${local.name}-backend"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_ecr_repository" "ui" {
  name = "${local.name}-ui"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_ecr_repository" "migrations" {
  name = "${local.name}-migrations"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}
