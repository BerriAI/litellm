resource "aws_ecr_repository" "gateway" {
  name                 = "${var.name}/litellm-rust-gateway"
  force_delete         = true
  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_repository" "tools" {
  name                 = "${var.name}/litellm-loadtest"
  force_delete         = true
  image_tag_mutability = "MUTABLE"
}

locals {
  registry      = split("/", aws_ecr_repository.gateway.repository_url)[0]
  gateway_image = var.gateway_image != "" ? var.gateway_image : "${aws_ecr_repository.gateway.repository_url}:${var.image_tag}"
  tools_image   = var.tools_image != "" ? var.tools_image : "${aws_ecr_repository.tools.repository_url}:${var.image_tag}"
}
