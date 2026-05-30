provider "aws" {
  region = var.region

  default_tags {
    tags = merge(
      {
        "litellm:stack" = local.name
        "managed-by"    = "terraform"
      },
      var.tags,
    )
  }
}
