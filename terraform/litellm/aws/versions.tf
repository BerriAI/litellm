terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

data "aws_region" "current" {}

# The module declares no provider, so `var.region` and the provider's region are
# two independent inputs that nothing forces to agree. Every resource is created
# wherever the provider points, while `var.region` is what gets interpolated into
# the log-group region, the rds-db:connect ARN, and S3_REGION_NAME, so a mismatch
# builds a stack whose ARNs name a region its resources are not in. A `check`
# rather than a precondition: the provider is the caller's to configure, and
# failing the plan of an existing stack over it would be a breaking change.
check "provider_region_matches_var_region" {
  assert {
    condition     = data.aws_region.current.name == var.region
    error_message = "The AWS provider is pointed at ${data.aws_region.current.name} while var.region is ${var.region}. Resources will be created in ${data.aws_region.current.name}, but the CloudWatch log configuration, the rds-db:connect ARN, and S3_REGION_NAME will all name ${var.region}. Set both to the same region, e.g. export AWS_REGION=${var.region}."
  }
}
